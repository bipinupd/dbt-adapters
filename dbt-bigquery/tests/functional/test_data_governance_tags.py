"""End-to-end functional tests for BigQuery column-level ``data_governance_tags``.

These tests run live against BigQuery using the active dbt test profile (e.g.
``pytest --profile oauth tests/functional/test_data_governance_tags.py -v -n 0``).

Required GCP context:
- The target BigQuery project must belong to a Google Cloud Organization so that
  Resource Manager ``DATA_GOVERNANCE`` tag keys (``purpose=DATA_GOVERNANCE``) can be
  attached to BigQuery columns.
- Existing tag keys/values can be supplied via environment variables:
  ``BIGQUERY_TEST_DGT_KEY`` (e.g. ``<project>/dbt_dgt_test_class``) and
  ``BIGQUERY_TEST_DGT_DOMAIN_KEY`` (e.g. ``<project>/dbt_dgt_test_domain``).
  If unset, the test fixture dynamically creates (and reuses idempotently) isolated
  ``DATA_GOVERNANCE`` tag keys in the target project and cleans up all transient
  taxonomies, V2 data policies, and dbt datasets on teardown. Set
  ``BIGQUERY_TEST_DGT_DELETE_KEYS=1`` to also delete the provisioned tag keys after
  the run.
"""

import copy
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

import google.auth
import google.auth.transport.requests
import pytest
from google.cloud import bigquery

from dbt.adapters.bigquery import BigQueryRelation
from dbt.tests.util import get_connection, relation_from_name, run_dbt, write_config_file

_CRM_V3_BASE = "https://cloudresourcemanager.googleapis.com/v3"
_DATAPOLICY_V2_BASE = "https://bigquerydatapolicy.googleapis.com/v2"
_DATACATALOG_V1_BASE = "https://datacatalog.googleapis.com/v1"

_DEFAULT_CLASS_KEY_SHORT = "dbt_dgt_e2e_class"
_DEFAULT_DOMAIN_KEY_SHORT = "dbt_dgt_e2e_domain"


# ---------------------------------------------------------------------------
# Minimal authenticated GCP REST helpers (zero extra package dependencies)
# ---------------------------------------------------------------------------


def _get_access_token_and_project() -> Tuple[str, Optional[str]]:
    creds, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    req = google.auth.transport.requests.Request()
    creds.refresh(req)
    return creds.token, project


def _gcp_rest(
    method: str,
    url: str,
    body: Optional[Dict[str, Any]] = None,
    *,
    allow_not_found: bool = False,
) -> Optional[Dict[str, Any]]:
    token, _ = _get_access_token_and_project()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        if allow_not_found and exc.code == 404:
            return None
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {err_body}") from exc


def _wait_for_crm_lro(op_data: Dict[str, Any], timeout_s: float = 60.0) -> Dict[str, Any]:
    if op_data.get("done"):
        if "error" in op_data:
            raise RuntimeError(f"CRM operation failed: {op_data['error']}")
        return op_data.get("response", op_data)
    op_name = op_data["name"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(1.0)
        status = _gcp_rest("GET", f"{_CRM_V3_BASE}/{op_name}")
        if status and status.get("done"):
            if "error" in status:
                raise RuntimeError(f"CRM operation {op_name} failed: {status['error']}")
            return status.get("response", status)
    raise TimeoutError(f"Timed out waiting for CRM operation {op_name}")


def _resolve_project_number(project_id: str) -> str:
    proj = _gcp_rest("GET", f"{_CRM_V3_BASE}/projects/{project_id}")
    name = proj["name"]  # "projects/<project_number>"
    return name.split("/", 1)[1]


def _ensure_dg_tag_key(project_number: str, short_name: str, description: str) -> str:
    resp = _gcp_rest(
        "GET", f"{_CRM_V3_BASE}/tagKeys?parent=projects/{project_number}&pageSize=300"
    )
    for item in resp.get("tagKeys", []):
        if item.get("shortName") == short_name:
            return item["name"]

    op = _gcp_rest(
        "POST",
        f"{_CRM_V3_BASE}/tagKeys",
        {
            "parent": f"projects/{project_number}",
            "shortName": short_name,
            "description": description,
            "purpose": "DATA_GOVERNANCE",
        },
    )
    res = _wait_for_crm_lro(op)
    return res["name"]


def _ensure_dg_tag_value(parent_resource: str, short_name: str, description: str) -> str:
    resp = _gcp_rest(
        "GET", f"{_CRM_V3_BASE}/tagValues?parent={parent_resource}&pageSize=300"
    )
    for item in resp.get("tagValues", []):
        if item.get("shortName") == short_name:
            return item["name"]

    op = _gcp_rest(
        "POST",
        f"{_CRM_V3_BASE}/tagValues",
        {
            "parent": parent_resource,
            "shortName": short_name,
            "description": description,
        },
    )
    res = _wait_for_crm_lro(op)
    return res["name"]


def _get_field_dict(fields: List[Dict[str, Any]], dotted_path: str) -> Dict[str, Any]:
    parts = dotted_path.split(".")
    current_fields = fields
    node: Optional[Dict[str, Any]] = None
    for part in parts:
        match = next((f for f in current_fields if f["name"] == part), None)
        assert match is not None, f"Column path '{dotted_path}' not found in schema: {fields}"
        node = match
        current_fields = match.get("fields", [])
    assert node is not None
    return node


def _fetch_raw_fields(project, model_name: str) -> List[Dict[str, Any]]:
    relation: BigQueryRelation = relation_from_name(project.adapter, model_name)
    with get_connection(project.adapter) as conn:
        bq_table = conn.handle.get_table(
            project.adapter.connections.table_ref(
                relation.database, relation.schema, relation.table
            )
        )
        return bq_table.to_api_repr()["schema"]["fields"]


def _fetch_info_schema_dg_tags(project, table_name: str) -> Dict[str, Dict[str, str]]:
    relation: BigQueryRelation = relation_from_name(project.adapter, table_name)
    sql = (
        f"SELECT field_path, data_governance_tags "
        f"FROM `{relation.database}`.`{relation.schema}`.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS "
        f"WHERE table_name = '{relation.table}'"
    )
    with get_connection(project.adapter) as conn:
        rows = list(conn.handle.query(sql).result())
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        tags = row["data_governance_tags"] or []
        result[row["field_path"]] = {entry["key"]: entry["value"] for entry in tags}
    return result


def _resolve_bq_location(project=None) -> str:
    """Resolve the active BigQuery dataset location in lowercase API form (defaults to 'us')."""
    raw = os.getenv("BIGQUERY_TEST_LOCATION")
    if not raw and project is not None:
        creds = getattr(getattr(project.adapter, "config", None), "credentials", None)
        raw = getattr(creds, "location", None)
    return (raw or "us").lower()


def _clear_field_tags_recursive(fields: List[Dict[str, Any]]) -> bool:
    """Clear dataGovernanceTagsInfo and policyTags on leaf fields in place.

    Returns True if any leaf field carried a tag that was cleared.
    """
    changed = False
    for field in fields:
        subfields = field.get("fields")
        if subfields:
            if _clear_field_tags_recursive(subfields):
                changed = True
            continue
        dgt_info = field.get("dataGovernanceTagsInfo")
        if isinstance(dgt_info, dict) and dgt_info.get("dataGovernanceTags"):
            field["dataGovernanceTagsInfo"] = {"dataGovernanceTags": {}}
            changed = True
        pt_info = field.get("policyTags")
        if isinstance(pt_info, dict) and pt_info.get("names"):
            field["policyTags"] = {"names": []}
            changed = True
    return changed


def _detach_dataset_tags(project) -> None:
    """Detach governance/policy tags from every TABLE in the test schema prior to drop.

    Prevents BigQuery's soft-delete/time-travel tag holds from blocking Resource Manager
    TagValue/TagKey deletion when ``BIGQUERY_TEST_DGT_DELETE_KEYS=1``.
    """
    with get_connection(project.adapter) as conn:
        client = conn.handle
        dataset_ref = f"{project.database}.{project.test_schema}"
        try:
            tables = list(client.list_tables(dataset_ref))
        except Exception:
            return
        for tbl_item in tables:
            if tbl_item.table_type != "TABLE":
                continue
            try:
                full_tbl = client.get_table(tbl_item.reference)
                raw_fields = copy.deepcopy(
                    full_tbl.to_api_repr().get("schema", {}).get("fields", [])
                )
                if _clear_field_tags_recursive(raw_fields):
                    patched = bigquery.Table(
                        full_tbl.reference,
                        schema=[bigquery.SchemaField.from_api_repr(f) for f in raw_fields],
                    )
                    client.update_table(patched, ["schema"])
            except Exception:
                pass


@pytest.fixture(scope="class", autouse=True)
def _detach_tags_on_teardown(project):
    yield
    _detach_dataset_tags(project)


def _get_active_principal_email(project) -> Optional[str]:
    try:
        creds, _ = google.auth.default()
        if getattr(creds, "service_account_email", None):
            sa = creds.service_account_email
            return f"principal://iam.googleapis.com/projects/-/serviceAccounts/{sa}"
        token, _ = _get_access_token_and_project()
        req = urllib.request.Request(
            f"https://oauth2.googleapis.com/tokeninfo?access_token={token}"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        email = info.get("email")
        if email:
            return f"principal://goog/subject/{email}"
    except Exception:
        pass
    return None


@pytest.fixture(scope="module")
def dg_tag_fixture() -> Dict[str, Any]:
    """Ensure Data Governance Tag Keys and hierarchical Tag Values exist in the test project."""
    project_id = (
        os.getenv("BIGQUERY_TEST_PROJECT")
        or _get_access_token_and_project()[1]
    )
    if not project_id:
        pytest.skip("No BigQuery GCP project configured for live data governance tag tests.")

    override_class = os.getenv("BIGQUERY_TEST_DGT_KEY")
    override_domain = os.getenv("BIGQUERY_TEST_DGT_DOMAIN_KEY")
    if override_class and override_domain:
        yield {
            "project_id": project_id,
            "class_key": override_class,
            "domain_key": override_domain,
            "values": {
                "pii": "pii",
                "email": "email",
                "ssn": "ssn",
                "public": "public",
                "finance": "finance",
            },
        }
        return

    try:
        project_number = _resolve_project_number(project_id)
        k_class = _ensure_dg_tag_key(
            project_number,
            _DEFAULT_CLASS_KEY_SHORT,
            "dbt-bigquery E2E test classification tag key",
        )
        v_pii = _ensure_dg_tag_value(k_class, "pii", "PII root tag value")
        v_email = _ensure_dg_tag_value(v_pii, "email", "Email child under PII")
        v_ssn = _ensure_dg_tag_value(v_pii, "ssn", "SSN child under PII")
        v_public = _ensure_dg_tag_value(k_class, "public", "Public non-sensitive leaf")

        k_domain = _ensure_dg_tag_key(
            project_number,
            _DEFAULT_DOMAIN_KEY_SHORT,
            "dbt-bigquery E2E test domain tag key",
        )
        v_finance = _ensure_dg_tag_value(k_domain, "finance", "Finance domain leaf")
    except Exception as exc:
        pytest.skip(f"Could not provision DATA_GOVERNANCE tags in '{project_id}': {exc}")

    fixture_info = {
        "project_id": project_id,
        "project_number": project_number,
        "class_key": f"{project_id}/{_DEFAULT_CLASS_KEY_SHORT}",
        "domain_key": f"{project_id}/{_DEFAULT_DOMAIN_KEY_SHORT}",
        "class_key_resource": k_class,
        "domain_key_resource": k_domain,
        "value_resources": [v_email, v_ssn, v_pii, v_public, v_finance],
        "values": {
            "pii": "pii",
            "email": "email",
            "ssn": "ssn",
            "public": "public",
            "finance": "finance",
        },
    }
    yield fixture_info

    if os.getenv("BIGQUERY_TEST_DGT_DELETE_KEYS") == "1":
        for tv in fixture_info["value_resources"]:
            try:
                op = _gcp_rest("DELETE", f"{_CRM_V3_BASE}/{tv}", allow_not_found=True)
                if op and "name" in op:
                    _wait_for_crm_lro(op)
            except Exception:
                pass
        for tk in (k_class, k_domain):
            try:
                op = _gcp_rest("DELETE", f"{_CRM_V3_BASE}/{tk}", allow_not_found=True)
                if op and "name" in op:
                    _wait_for_crm_lro(op)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 1. Table model: top-level columns + Jinja var() + descriptions + INFO_SCHEMA
# ---------------------------------------------------------------------------


_TABLE_MODEL_SQL = """{{
  config(
    materialized='table',
    persist_docs={ 'columns': true }
  )
}}

select
  1 as id,
  'alice@example.com' as email,
  '123-45-6789' as ssn,
  'gold' as tier
"""


class TestBigQueryDataGovernanceTagsTable:
    """Attach Data Governance tags to top-level table columns alongside descriptions."""

    @pytest.fixture(scope="class")
    def project_config_update(self, dg_tag_fixture):
        return {
            "config-version": 2,
            "vars": {
                "dgt_class_key": dg_tag_fixture["class_key"],
                "dgt_email_val": dg_tag_fixture["values"]["email"],
                "dgt_ssn_val": dg_tag_fixture["values"]["ssn"],
            },
        }

    @pytest.fixture(scope="class")
    def models(self, dg_tag_fixture):
        class_key = dg_tag_fixture["class_key"]
        schema_yml = f"""version: 2

models:
  - name: dgt_customers
    columns:
      - name: id
        description: "Primary identifier (untagged)"
      - name: email
        description: "Customer email address"
        data_governance_tags:
          {class_key}: "{{{{ var('dgt_email_val') }}}}"
      - name: ssn
        description: "US SSN"
        data_governance_tags: "{{{{ {{var('dgt_class_key'): var('dgt_ssn_val')}} }}}}"
      - name: tier
        description: "Public loyalty tier"
        data_governance_tags:
          {class_key}: public
"""
        return {
            "dgt_customers.sql": _TABLE_MODEL_SQL,
            "schema.yml": schema_yml,
        }

    def test_table_columns_and_info_schema(self, project, dg_tag_fixture):
        results = run_dbt(["run", "--select", "dgt_customers"])
        assert len(results) == 1
        assert results[0].status == "success"

        class_key = dg_tag_fixture["class_key"]
        raw_fields = _fetch_raw_fields(project, "dgt_customers")

        id_field = _get_field_dict(raw_fields, "id")
        assert id_field.get("description") == "Primary identifier (untagged)"
        assert "dataGovernanceTagsInfo" not in id_field

        email_field = _get_field_dict(raw_fields, "email")
        assert email_field.get("description") == "Customer email address"
        assert email_field.get("dataGovernanceTagsInfo") == {
            "dataGovernanceTags": {class_key: "email"}
        }

        ssn_field = _get_field_dict(raw_fields, "ssn")
        assert ssn_field.get("description") == "US SSN"
        assert ssn_field.get("dataGovernanceTagsInfo") == {
            "dataGovernanceTags": {class_key: "ssn"}
        }

        tier_field = _get_field_dict(raw_fields, "tier")
        assert tier_field.get("dataGovernanceTagsInfo") == {
            "dataGovernanceTags": {class_key: "public"}
        }

        # Verify SQL INFORMATION_SCHEMA.COLUMN_FIELD_PATHS visibility
        info_tags = _fetch_info_schema_dg_tags(project, "dgt_customers")
        assert info_tags["id"] == {}
        assert info_tags["email"] == {class_key: "email"}
        assert info_tags["ssn"] == {class_key: "ssn"}
        assert info_tags["tier"] == {class_key: "public"}


# ---------------------------------------------------------------------------
# 2. Nested STRUCT / multi-level STRUCT / ARRAY<STRUCT> leaf fields
# ---------------------------------------------------------------------------


_NESTED_MODEL_SQL = """{{
  config(
    materialized='incremental',
    persist_docs={ 'columns': true }
  )
}}

select
  1 as customer_id,
  struct(
    'alice.work@example.com' as work_email,
    struct('999-00-1111' as ssn, 'US' as country) as identity
  ) as profile,
  [
    struct('actor@example.com' as actor_email, 'login' as event_type)
  ] as events
"""

_NESTED_YML_INITIAL = """version: 2

models:
  - name: dgt_nested_customers
    columns:
      - name: profile
        description: "Nested customer profile struct"
      - name: profile.work_email
        description: "Work email inside struct"
        data_governance_tags: "{{ {var('dgt_class_key'): 'email'} }}"
      - name: profile.identity
        description: "Nested identity record"
      - name: profile.identity.ssn
        description: "Deeply nested SSN leaf"
        data_governance_tags: "{{ {var('dgt_class_key'): 'ssn'} }}"
      - name: profile.identity.country
        description: "Public country code"
      - name: events.actor_email
        description: "Repeated ARRAY<STRUCT> leaf email"
        data_governance_tags: "{{ {var('dgt_domain_key'): 'finance'} }}"
"""

_NESTED_YML_UPDATED = """version: 2

models:
  - name: dgt_nested_customers
    columns:
      - name: profile
        description: "Nested customer profile struct"
      - name: profile.work_email
        description: "Work email tag updated to pii"
        data_governance_tags: "{{ {var('dgt_class_key'): 'pii'} }}"
      - name: profile.identity.ssn
        description: "Deeply nested SSN tag removed"
      - name: events.actor_email
        description: "Repeated leaf tag cleared explicitly"
        data_governance_tags: {}
"""


class TestBigQueryDataGovernanceTagsNestedStruct:
    """Attach, update, and remove tags on nested STRUCT and ARRAY<STRUCT> leaf fields."""

    @pytest.fixture(scope="class")
    def project_config_update(self, dg_tag_fixture):
        return {
            "config-version": 2,
            "vars": {
                "dgt_class_key": dg_tag_fixture["class_key"],
                "dgt_domain_key": dg_tag_fixture["domain_key"],
            },
        }

    @pytest.fixture(scope="class")
    def models(self):
        return {
            "dgt_nested_customers.sql": _NESTED_MODEL_SQL,
            "schema.yml": _NESTED_YML_INITIAL,
        }

    def test_nested_struct_and_repeated_record_lifecycle(self, project, dg_tag_fixture):
        class_key = dg_tag_fixture["class_key"]
        domain_key = dg_tag_fixture["domain_key"]

        res1 = run_dbt(["run", "--select", "dgt_nested_customers"])
        assert len(res1) == 1 and res1[0].status == "success"

        raw1 = _fetch_raw_fields(project, "dgt_nested_customers")
        assert _get_field_dict(raw1, "profile").get("description") == "Nested customer profile struct"
        assert "dataGovernanceTagsInfo" not in _get_field_dict(raw1, "profile")
        assert _get_field_dict(raw1, "profile.work_email")["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {class_key: "email"}
        }
        assert _get_field_dict(raw1, "profile.identity.ssn")["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {class_key: "ssn"}
        }
        assert "dataGovernanceTagsInfo" not in _get_field_dict(raw1, "profile.identity.country")
        assert _get_field_dict(raw1, "events.actor_email")["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {domain_key: "finance"}
        }

        info1 = _fetch_info_schema_dg_tags(project, "dgt_nested_customers")
        assert info1["profile.work_email"] == {class_key: "email"}
        assert info1["profile.identity.ssn"] == {class_key: "ssn"}
        assert info1["events.actor_email"] == {domain_key: "finance"}

        # Incremental update WITHOUT --full-refresh: update one leaf, clear two leaves
        write_config_file(
            _NESTED_YML_UPDATED, project.project_root + "/models", "schema.yml"
        )
        res2 = run_dbt(["run", "--select", "dgt_nested_customers"])
        assert len(res2) == 1 and res2[0].status == "success"

        info2 = _fetch_info_schema_dg_tags(project, "dgt_nested_customers")
        assert info2["profile.work_email"] == {class_key: "pii"}
        assert info2["profile.identity.ssn"] == {}
        assert info2["events.actor_email"] == {}


# ---------------------------------------------------------------------------
# 3. Incremental table full lifecycle (attach -> replace value -> switch key
#    -> implicit delete -> explicit {} delete -> untouched sibling preserved)
# ---------------------------------------------------------------------------


_INC_MODEL_SQL = """{{
  config(
    materialized='incremental',
    persist_docs={ 'columns': true }
  )
}}

select
  1 as id,
  'bob@example.com' as email,
  'SEC-100' as secret_code,
  'ext-val' as external_col
"""


class TestBigQueryDataGovernanceTagsIncrementalLifecycle:
    """Full stateful lifecycle on an incremental model without full-refresh."""

    @pytest.fixture(scope="class")
    def project_config_update(self, dg_tag_fixture):
        return {
            "config-version": 2,
            "vars": {
                "dgt_class_key": dg_tag_fixture["class_key"],
                "dgt_domain_key": dg_tag_fixture["domain_key"],
            },
        }

    @pytest.fixture(scope="class")
    def models(self):
        initial_yml = """version: 2
models:
  - name: dgt_incremental
    columns:
      - name: email
        description: "Initial description"
        data_governance_tags: "{{ {var('dgt_class_key'): 'public'} }}"
      - name: secret_code
        description: "Initial secret"
        data_governance_tags: "{{ {var('dgt_class_key'): 'ssn'} }}"
"""
        return {
            "dgt_incremental.sql": _INC_MODEL_SQL,
            "schema.yml": initial_yml,
        }

    def test_incremental_attach_replace_switch_clear_preserve(self, project, dg_tag_fixture):
        class_key = dg_tag_fixture["class_key"]
        domain_key = dg_tag_fixture["domain_key"]
        schema_dir = project.project_root + "/models"

        # Step 1: Initial creation
        run_dbt(["run", "--select", "dgt_incremental"])
        info = _fetch_info_schema_dg_tags(project, "dgt_incremental")
        assert info["email"] == {class_key: "public"}
        assert info["secret_code"] == {class_key: "ssn"}

        # Attach an out-of-band tag to `external_col` (NOT listed in schema.yml)
        rel = relation_from_name(project.adapter, "dgt_incremental")
        with get_connection(project.adapter) as conn:
            t_ref = project.adapter.connections.table_ref(rel.database, rel.schema, rel.table)
            tbl = conn.handle.get_table(t_ref)
            raw_fields = tbl.to_api_repr()["schema"]["fields"]
            for field in raw_fields:
                if field["name"] == "external_col":
                    field["dataGovernanceTagsInfo"] = {
                        "dataGovernanceTags": {domain_key: "finance"}
                    }
            new_tbl = bigquery.Table(
                t_ref,
                schema=[bigquery.SchemaField.from_api_repr(f) for f in raw_fields],
            )
            conn.handle.update_table(new_tbl, ["schema"])

        # Step 2: Update `email` value (public -> email) and switch `secret_code` key (class -> domain)
        write_config_file(
            """version: 2
models:
  - name: dgt_incremental
    columns:
      - name: email
        description: "Updated email desc"
        data_governance_tags: "{{ {var('dgt_class_key'): 'email'} }}"
      - name: secret_code
        description: "Switched tag key"
        data_governance_tags: "{{ {var('dgt_domain_key'): 'finance'} }}"
""",
            schema_dir,
            "schema.yml",
        )
        run_dbt(["run", "--select", "dgt_incremental"])
        info = _fetch_info_schema_dg_tags(project, "dgt_incremental")
        assert info["email"] == {class_key: "email"}
        assert info["secret_code"] == {domain_key: "finance"}
        # Unlisted column kept its out-of-band tag
        assert info["external_col"] == {domain_key: "finance"}

        # Step 3: Implicitly remove `email` tag (omit property) & explicitly clear `secret_code` with {}
        write_config_file(
            """version: 2
models:
  - name: dgt_incremental
    columns:
      - name: email
        description: "Description retained after tag removal"
      - name: secret_code
        description: "Explicit empty dict clear"
        data_governance_tags: {}
""",
            schema_dir,
            "schema.yml",
        )
        run_dbt(["run", "--select", "dgt_incremental"])
        raw = _fetch_raw_fields(project, "dgt_incremental")
        assert _get_field_dict(raw, "email")["description"] == "Description retained after tag removal"
        assert "dataGovernanceTagsInfo" not in _get_field_dict(raw, "email")
        assert "dataGovernanceTagsInfo" not in _get_field_dict(raw, "secret_code")
        # Out-of-band sibling tag is STILL preserved!
        assert _get_field_dict(raw, "external_col")["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {domain_key: "finance"}
        }


# ---------------------------------------------------------------------------
# 4. Atomic migration swap: legacy policy_tags <-> data_governance_tags
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def temp_policy_tag(dg_tag_fixture):
    override = os.getenv("BIGQUERY_TEST_POLICY_TAG")
    if override:
        yield override
        return

    project_id = dg_tag_fixture["project_id"]
    location = _resolve_bq_location()
    parent = f"projects/{project_id}/locations/{location}"
    suffix = uuid.uuid4().hex[:8]
    taxonomy_name = None
    try:
        tax = _gcp_rest(
            "POST",
            f"{_DATACATALOG_V1_BASE}/{parent}/taxonomies",
            {
                "displayName": f"dbt_dgt_e2e_tmp_tax_{suffix}",
                "description": "Temporary taxonomy for dbt atomic PT<->DGT swap test",
            },
        )
        taxonomy_name = tax["name"]
        tag = _gcp_rest(
            "POST",
            f"{_DATACATALOG_V1_BASE}/{taxonomy_name}/policyTags",
            {"displayName": "tmp_pii_email", "description": "Temp test policy tag"},
        )
        policy_tag_name = tag["name"]
    except Exception as exc:
        pytest.skip(f"Could not create temporary policy tag in '{project_id}': {exc}")

    try:
        yield policy_tag_name
    finally:
        if taxonomy_name:
            try:
                _gcp_rest("DELETE", f"{_DATACATALOG_V1_BASE}/{taxonomy_name}", allow_not_found=True)
            except Exception:
                pass


class TestBigQueryDataGovernanceTagsAtomicSwap:
    """Atomically migrate an incremental model from policy_tags to data_governance_tags and back."""

    @pytest.fixture(scope="class")
    def project_config_update(self, dg_tag_fixture, temp_policy_tag):
        return {
            "config-version": 2,
            "vars": {
                "legacy_policy_tag": temp_policy_tag,
                "dgt_class_key": dg_tag_fixture["class_key"],
            },
        }

    @pytest.fixture(scope="class")
    def models(self):
        return {
            "dgt_migration_swap.sql": """{{
  config(
    materialized='incremental',
    persist_docs={ 'columns': true }
  )
}}
select
  'alice@example.com' as email,
  struct('123-45-6789' as ssn) as identity
""",
            "schema.yml": """version: 2
models:
  - name: dgt_migration_swap
    columns:
      - name: email
        description: "Legacy policy tag"
        policy_tags:
          - '{{ var("legacy_policy_tag") }}'
      - name: identity.ssn
        description: "Nested legacy policy tag"
        policy_tags:
          - '{{ var("legacy_policy_tag") }}'
""",
        }

    def test_atomic_pt_to_dgt_and_rollback(self, project, dg_tag_fixture, temp_policy_tag):
        class_key = dg_tag_fixture["class_key"]
        schema_dir = project.project_root + "/models"

        # 1. Start with legacy Data Catalog policy_tags
        run_dbt(["run", "--select", "dgt_migration_swap"])
        raw1 = _fetch_raw_fields(project, "dgt_migration_swap")
        assert _get_field_dict(raw1, "email")["policyTags"]["names"] == [temp_policy_tag]
        assert "dataGovernanceTagsInfo" not in _get_field_dict(raw1, "email")
        assert _get_field_dict(raw1, "identity.ssn")["policyTags"]["names"] == [temp_policy_tag]

        # 2. Forward cutover: swap BOTH top-level & nested leaf atomically to data_governance_tags
        write_config_file(
            """version: 2
models:
  - name: dgt_migration_swap
    columns:
      - name: email
        description: "Migrated to data governance tag"
        data_governance_tags: "{{ {var('dgt_class_key'): 'email'} }}"
      - name: identity.ssn
        description: "Nested migrated to data governance tag"
        data_governance_tags: "{{ {var('dgt_class_key'): 'ssn'} }}"
""",
            schema_dir,
            "schema.yml",
        )
        run_dbt(["run", "--select", "dgt_migration_swap"])
        raw2 = _fetch_raw_fields(project, "dgt_migration_swap")
        assert "policyTags" not in _get_field_dict(raw2, "email")
        assert _get_field_dict(raw2, "email")["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {class_key: "email"}
        }
        assert "policyTags" not in _get_field_dict(raw2, "identity.ssn")
        assert _get_field_dict(raw2, "identity.ssn")["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {class_key: "ssn"}
        }

        # 3. Rollback: swap BOTH columns back atomically to legacy policy_tags
        write_config_file(
            """version: 2
models:
  - name: dgt_migration_swap
    columns:
      - name: email
        description: "Rolled back to policy tag"
        policy_tags:
          - '{{ var("legacy_policy_tag") }}'
      - name: identity.ssn
        description: "Nested rolled back to policy tag"
        policy_tags:
          - '{{ var("legacy_policy_tag") }}'
""",
            schema_dir,
            "schema.yml",
        )
        run_dbt(["run", "--select", "dgt_migration_swap"])
        raw3 = _fetch_raw_fields(project, "dgt_migration_swap")
        assert _get_field_dict(raw3, "email")["policyTags"]["names"] == [temp_policy_tag]
        assert "dataGovernanceTagsInfo" not in _get_field_dict(raw3, "email")
        assert _get_field_dict(raw3, "identity.ssn")["policyTags"]["names"] == [temp_policy_tag]
        assert "dataGovernanceTagsInfo" not in _get_field_dict(raw3, "identity.ssn")


# ---------------------------------------------------------------------------
# 5. Seed & Snapshot models
# ---------------------------------------------------------------------------


class TestBigQueryDataGovernanceTagsSeed:
    """Verify `dbt seed` applies `data_governance_tags` when persist_docs.columns=true."""

    @pytest.fixture(scope="class")
    def project_config_update(self, dg_tag_fixture):
        return {
            "config-version": 2,
            "vars": {"dgt_class_key": dg_tag_fixture["class_key"]},
            "seeds": {"test": {"+persist_docs": {"columns": True}}},
        }

    @pytest.fixture(scope="class")
    def seeds(self):
        return {
            "dgt_seed_users.csv": "id,email\n1,alice@example.com\n2,bob@example.com\n",
            "properties.yml": """version: 2
seeds:
  - name: dgt_seed_users
    columns:
      - name: id
        description: "Seed user ID"
      - name: email
        description: "Seed user email"
        data_governance_tags: "{{ {var('dgt_class_key'): 'email'} }}"
""",
        }

    def test_seed_applies_data_governance_tag(self, project, dg_tag_fixture):
        results = run_dbt(["seed"])
        assert len(results) == 1 and results[0].status == "success"
        raw = _fetch_raw_fields(project, "dgt_seed_users")
        assert _get_field_dict(raw, "email")["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {dg_tag_fixture["class_key"]: "email"}
        }
        assert "dataGovernanceTagsInfo" not in _get_field_dict(raw, "id")


class TestBigQueryDataGovernanceTagsSnapshot:
    """Verify `dbt snapshot` applies `data_governance_tags` on the snapshot table."""

    @pytest.fixture(scope="class")
    def project_config_update(self, dg_tag_fixture):
        return {
            "config-version": 2,
            "vars": {"dgt_class_key": dg_tag_fixture["class_key"]},
        }

    @pytest.fixture(scope="class")
    def models(self):
        return {
            "dgt_snap_src.sql": """{{ config(materialized='table') }}
select 1 as id, 'carol@example.com' as email, timestamp('2026-01-01 00:00:00 UTC') as updated_at
"""
        }

    @pytest.fixture(scope="class")
    def snapshots(self):
        return {
            "dgt_users_snapshot.sql": """{% snapshot dgt_users_snapshot %}
{{
  config(
    target_schema=target.schema,
    unique_key='id',
    strategy='timestamp',
    updated_at='updated_at',
    persist_docs={'columns': true}
  )
}}
select id, email, updated_at from {{ ref('dgt_snap_src') }}
{% endsnapshot %}
""",
            "schema.yml": """version: 2
snapshots:
  - name: dgt_users_snapshot
    columns:
      - name: email
        description: "Snapshotted email"
        data_governance_tags: "{{ {var('dgt_class_key'): 'email'} }}"
""",
        }

    def test_snapshot_applies_data_governance_tag(self, project, dg_tag_fixture):
        run_dbt(["run", "--select", "dgt_snap_src"])
        results = run_dbt(["snapshot"])
        assert len(results) == 1 and results[0].status == "success"
        info = _fetch_info_schema_dg_tags(project, "dgt_users_snapshot")
        assert info["email"] == {dg_tag_fixture["class_key"]: "email"}


# ---------------------------------------------------------------------------
# 6. Contracted model (`contract: {enforced: true}`)
# ---------------------------------------------------------------------------


class TestBigQueryDataGovernanceTagsContractModel:
    """Verify `contract: {enforced: true}` models with nested STRUCT + DGT."""

    @pytest.fixture(scope="class")
    def project_config_update(self, dg_tag_fixture):
        return {
            "config-version": 2,
            "vars": {"dgt_class_key": dg_tag_fixture["class_key"]},
        }

    @pytest.fixture(scope="class")
    def models(self):
        return {
            "dgt_contract_model.sql": """{{
  config(
    materialized='table',
    persist_docs={ 'columns': true }
  )
}}
select
  cast(1 as int64) as id,
  'dana@example.com' as email,
  struct('999-11-2222' as ssn) as details
""",
            "schema.yml": """version: 2
models:
  - name: dgt_contract_model
    config:
      contract:
        enforced: true
    columns:
      - name: id
        data_type: int64
        description: "ID column"
      - name: email
        data_type: string
        description: "Email with contract + DGT"
        data_governance_tags: "{{ {var('dgt_class_key'): 'email'} }}"
      - name: details.ssn
        data_type: string
        description: "Contracted nested leaf SSN"
        data_governance_tags: "{{ {var('dgt_class_key'): 'ssn'} }}"
""",
        }

    def test_contracted_model_attaches_tags(self, project, dg_tag_fixture):
        class_key = dg_tag_fixture["class_key"]
        results = run_dbt(["run", "--select", "dgt_contract_model"])
        assert len(results) == 1 and results[0].status == "success"
        info = _fetch_info_schema_dg_tags(project, "dgt_contract_model")
        assert info["email"] == {class_key: "email"}
        assert info["details.ssn"] == {class_key: "ssn"}


# ---------------------------------------------------------------------------
# 7. Negative / validation tests (views, STRUCT parents, mutual exclusivity,
#    hierarchical slash paths, multi-tag server limit)
# ---------------------------------------------------------------------------


class TestBigQueryDataGovernanceTagsValidation:
    """Live error-path tests confirming clear failure messages & safe view handling."""

    @pytest.fixture(scope="class")
    def project_config_update(self, dg_tag_fixture):
        return {
            "config-version": 2,
            "vars": {
                "dgt_class_key": dg_tag_fixture["class_key"],
                "dgt_domain_key": dg_tag_fixture["domain_key"],
            },
        }

    @pytest.fixture(scope="class")
    def models(self):
        return {
            "dgt_valid_view.sql": "{{ config(materialized='view', persist_docs={'columns': true}) }} select 'a@b.com' as email",
            "dgt_bad_view.sql": "{{ config(materialized='view', persist_docs={'columns': true}) }} select 'a@b.com' as email",
            "dgt_bad_struct.sql": "{{ config(materialized='table', persist_docs={'columns': true}) }} select struct('a@b.com' as email) as profile",
            "dgt_bad_both.sql": "{{ config(materialized='table', persist_docs={'columns': true}) }} select 'a@b.com' as email",
            "dgt_bad_path.sql": "{{ config(materialized='table', persist_docs={'columns': true}) }} select 'a@b.com' as email",
            "dgt_multi_tag.sql": "{{ config(materialized='table', persist_docs={'columns': true}) }} select 'a@b.com' as email",
            "schema.yml": """version: 2
models:
  - name: dgt_valid_view
    columns:
      - name: email
        description: "View column description only"
  - name: dgt_bad_view
    columns:
      - name: email
        data_governance_tags: "{{ {var('dgt_class_key'): 'email'} }}"
  - name: dgt_bad_struct
    columns:
      - name: profile
        data_governance_tags: "{{ {var('dgt_class_key'): 'pii'} }}"
  - name: dgt_bad_both
    columns:
      - name: email
        policy_tags:
          - projects/dummy/locations/us/taxonomies/1/policyTags/2
        data_governance_tags: "{{ {var('dgt_class_key'): 'email'} }}"
  - name: dgt_bad_path
    columns:
      - name: email
        data_governance_tags: "{{ {var('dgt_class_key'): 'pii/email'} }}"
  - name: dgt_multi_tag
    columns:
      - name: email
        data_governance_tags: "{{ {var('dgt_class_key'): 'public', var('dgt_domain_key'): 'finance'} }}"
""",
        }

    def test_valid_view_description_only_succeeds(self, project):
        res = run_dbt(["run", "--select", "dgt_valid_view"])
        assert len(res) == 1 and res[0].status == "success"
        raw = _fetch_raw_fields(project, "dgt_valid_view")
        assert _get_field_dict(raw, "email")["description"] == "View column description only"

    def test_view_with_dgt_fails_clearly(self, project):
        res = run_dbt(["run", "--select", "dgt_bad_view"], expect_pass=False)
        assert res[0].status == "error"
        assert "only supports on table columns" in (res[0].message or "")

    def test_struct_parent_with_dgt_fails_clearly(self, project):
        res = run_dbt(["run", "--select", "dgt_bad_struct"], expect_pass=False)
        assert res[0].status == "error"
        assert "RECORD/STRUCT" in (res[0].message or "")

    def test_both_policy_and_dgt_fails_clearly(self, project):
        res = run_dbt(["run", "--select", "dgt_bad_both"], expect_pass=False)
        assert res[0].status == "error"
        assert "cannot specify both" in (res[0].message or "")

    def test_hierarchical_path_value_fails_clearly(self, project):
        res = run_dbt(["run", "--select", "dgt_bad_path"], expect_pass=False)
        assert res[0].status == "error"
        assert "leaf tag value short name" in (res[0].message or "")

    def test_multi_tag_forwards_to_bigquery_server_quota(self, project):
        res = run_dbt(["run", "--select", "dgt_multi_tag"], expect_pass=False)
        assert res[0].status == "error"
        assert "Too many data governance tags" in (res[0].message or "")


# ---------------------------------------------------------------------------
# 8. Live BigQuery Data Policy API v2 binding & dynamic masking verification
# ---------------------------------------------------------------------------


class TestBigQueryDataGovernanceTagsDataPolicyV2:
    """Verify creating a V2 Data Policy bound to the dbt-attached Data Governance tag."""

    @pytest.fixture(scope="class")
    def project_config_update(self, dg_tag_fixture):
        return {
            "config-version": 2,
            "vars": {"dgt_class_key": dg_tag_fixture["class_key"]},
        }

    @pytest.fixture(scope="class")
    def models(self):
        return {
            "dgt_masked_customers.sql": """{{
  config(
    materialized='table',
    persist_docs={ 'columns': true }
  )
}}
select
  1 as customer_id,
  'alice.secret@example.com' as email,
  '123-45-6789' as ssn
""",
            "schema.yml": """version: 2
models:
  - name: dgt_masked_customers
    columns:
      - name: customer_id
        description: "Public customer identifier"
      - name: email
        description: "Email protected by V2 EMAIL_MASK data policy"
        data_governance_tags: "{{ {var('dgt_class_key'): 'email'} }}"
      - name: ssn
        description: "SSN protected by V2 SHA256 data policy"
        data_governance_tags: "{{ {var('dgt_class_key'): 'ssn'} }}"
""",
        }

    def test_data_policy_v2_binding_on_dbt_tagged_columns(self, project, dg_tag_fixture):
        project_id = dg_tag_fixture["project_id"]
        class_key = dg_tag_fixture["class_key"]
        location = _resolve_bq_location(project)
        parent = f"projects/{project_id}/locations/{location}"
        suffix = uuid.uuid4().hex[:8]
        email_policy_id = f"dbt_e2e_email_mask_{suffix}"
        ssn_policy_id = f"dbt_e2e_ssn_sha256_{suffix}"
        created_policies: List[str] = []

        principal = _get_active_principal_email(project)
        grantees = [principal] if principal else []

        try:
            # Create V2 DATA_MASKING_POLICY bound to `class_key` -> `email` (`EMAIL_MASK`)
            email_resp = _gcp_rest(
                "POST",
                f"{_DATAPOLICY_V2_BASE}/{parent}/dataPolicies",
                {
                    "dataPolicyId": email_policy_id,
                    "dataPolicy": {
                        "dataPolicyType": "DATA_MASKING_POLICY",
                        "dataGovernanceTag": {"key": class_key, "value": "email"},
                        "dataMaskingPolicy": {"predefinedExpression": "EMAIL_MASK"},
                        "grantees": grantees,
                    },
                },
            )
            created_policies.append(email_resp["name"])

            # Create V2 DATA_MASKING_POLICY bound to `class_key` -> `ssn` (`SHA256`)
            ssn_resp = _gcp_rest(
                "POST",
                f"{_DATAPOLICY_V2_BASE}/{parent}/dataPolicies",
                {
                    "dataPolicyId": ssn_policy_id,
                    "dataPolicy": {
                        "dataPolicyType": "DATA_MASKING_POLICY",
                        "dataGovernanceTag": {"key": class_key, "value": "ssn"},
                        "dataMaskingPolicy": {"predefinedExpression": "SHA256"},
                        "grantees": grantees,
                    },
                },
            )
            created_policies.append(ssn_resp["name"])

            # Run dbt to build table and attach Data Governance tags
            results = run_dbt(["run", "--select", "dgt_masked_customers"])
            assert len(results) == 1 and results[0].status == "success"

            # Verify live table carries the exact tags bound by the V2 data policies
            info = _fetch_info_schema_dg_tags(project, "dgt_masked_customers")
            assert info["email"] == {class_key: "email"}
            assert info["ssn"] == {class_key: "ssn"}

            # Verify both V2 Data Policies point at the matching tag key & leaf value
            fetched_email = _gcp_rest("GET", f"{_DATAPOLICY_V2_BASE}/{email_resp['name']}")
            assert fetched_email["dataPolicyType"] == "DATA_MASKING_POLICY"
            assert fetched_email["dataMaskingPolicy"]["predefinedExpression"] == "EMAIL_MASK"
            assert fetched_email["dataGovernanceTag"]["value"] == "email"

            fetched_ssn = _gcp_rest("GET", f"{_DATAPOLICY_V2_BASE}/{ssn_resp['name']}")
            assert fetched_ssn["dataPolicyType"] == "DATA_MASKING_POLICY"
            assert fetched_ssn["dataMaskingPolicy"]["predefinedExpression"] == "SHA256"
            assert fetched_ssn["dataGovernanceTag"]["value"] == "ssn"

            # Confirm the table is queryable (note: IAM Data Policy v2 enforcement can take
            # a few minutes to propagate for brand-new policies; if propagated, email is masked).
            rel = relation_from_name(project.adapter, "dgt_masked_customers")
            with get_connection(project.adapter) as conn:
                cfg = bigquery.QueryJobConfig(use_query_cache=False)
                row = list(
                    conn.handle.query(
                        f"SELECT customer_id, email, ssn FROM `{rel.database}`.`{rel.schema}`.`{rel.table}`",
                        job_config=cfg,
                    ).result()
                )[0]
                assert row["customer_id"] == 1
        finally:
            for pol_name in created_policies:
                try:
                    _gcp_rest("DELETE", f"{_DATAPOLICY_V2_BASE}/{pol_name}", allow_not_found=True)
                except Exception:
                    pass

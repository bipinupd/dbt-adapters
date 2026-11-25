from unittest.mock import MagicMock

import pytest
from google.cloud.bigquery import SchemaField, Table

import dbt_common.exceptions
from dbt.adapters.bigquery import BigQueryRelation
from dbt.adapters.bigquery.impl import BigQueryAdapter
from dbt.adapters.base import RelationType

from .test_bigquery_adapter import BaseTestBigQueryAdapter


def _build_mock_bq_table(
    project: str,
    dataset: str,
    table_name: str,
    fields: list,
    table_type: str = "TABLE",
) -> Table:
    api_repr = {
        "tableReference": {
            "projectId": project,
            "datasetId": dataset,
            "tableId": table_name,
        },
        "type": table_type,
        "schema": {"fields": fields},
    }
    return Table.from_api_repr(api_repr)


def _wire_fields(updated_table: Table) -> list:
    return updated_table._build_resource(["schema"])["schema"]["fields"]


class TestBigQueryDataGovernanceTagsUnit(BaseTestBigQueryAdapter):
    """Unit tests for column-level `data_governance_tags` support in BigQueryAdapter."""

    def _run_update_columns(
        self,
        initial_fields: list,
        dbt_columns: dict,
        *,
        table_type: str = "TABLE",
        relation_type: RelationType = RelationType.Table,
    ):
        bq_table = _build_mock_bq_table(
            "dbt-unit-000000", "dummy_schema", "customers", initial_fields, table_type=table_type
        )

        mock_client = MagicMock()
        mock_client.get_table.return_value = bq_table

        mock_conn = MagicMock()
        mock_conn.handle = mock_client

        relation = BigQueryRelation.create(
            database="dbt-unit-000000",
            schema="dummy_schema",
            identifier="customers",
            type=relation_type,
        )

        adapter = self.get_adapter("oauth")
        adapter.connections.get_thread_connection = MagicMock(return_value=mock_conn)
        adapter.update_columns(relation, dbt_columns)
        return mock_client

    def test_attach_single_top_level_tag_with_description(self):
        fields = [
            {"name": "id", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "email", "type": "STRING", "mode": "NULLABLE"},
        ]
        columns = {
            "id": {"name": "id", "description": "Primary key"},
            "email": {
                "name": "email",
                "description": "Customer email",
                "data_governance_tags": {"my-project/data_class": "email"},
            },
        }

        client = self._run_update_columns(fields, columns)
        client.update_table.assert_called_once()
        updated_table, update_fields = client.update_table.call_args.args
        assert update_fields == ["schema"]

        wire = _wire_fields(updated_table)
        id_field, email_field = wire[0], wire[1]

        assert id_field["description"] == "Primary key"
        assert id_field["policyTags"] == {"names": []}
        assert "dataGovernanceTagsInfo" not in id_field

        assert email_field["description"] == "Customer email"
        assert email_field["policyTags"] == {"names": []}
        assert email_field["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "email"}
        }

    def test_attach_org_and_numeric_project_scoped_keys(self):
        fields = [
            {"name": "org_col", "type": "STRING"},
            {"name": "proj_num_col", "type": "STRING"},
        ]
        columns = {
            "org_col": {
                "name": "org_col",
                "data_governance_tags": {"94408448616/corp.classification-v1": "restricted_pii"},
            },
            "proj_num_col": {
                "name": "proj_num_col",
                "data_governance_tags": {"609037062066/domain_tags": "finance-v2"},
            },
        }

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"94408448616/corp.classification-v1": "restricted_pii"}
        }
        assert wire[1]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"609037062066/domain_tags": "finance-v2"}
        }

    def test_replace_existing_tag_value_and_key(self):
        fields = [
            {
                "name": "email",
                "type": "STRING",
                "dataGovernanceTagsInfo": {
                    "dataGovernanceTags": {"my-project/data_class": "public"}
                },
            },
            {
                "name": "account_id",
                "type": "STRING",
                "dataGovernanceTagsInfo": {
                    "dataGovernanceTags": {"my-project/old_key": "internal"}
                },
            },
        ]
        columns = {
            "email": {
                "name": "email",
                "data_governance_tags": {"my-project/data_class": "email"},
            },
            "account_id": {
                "name": "account_id",
                "data_governance_tags": {"my-project/new_key": "finance"},
            },
        }

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "email"}
        }
        assert wire[1]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/new_key": "finance"}
        }

    def test_remove_tag_when_omitted_from_yaml(self):
        fields = [
            {
                "name": "email",
                "type": "STRING",
                "dataGovernanceTagsInfo": {
                    "dataGovernanceTags": {"my-project/data_class": "email"}
                },
            },
        ]
        # Column remains listed in schema.yml, but `data_governance_tags` was removed.
        columns = {"email": {"name": "email", "description": "Now untagged"}}

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["description"] == "Now untagged"
        assert wire[0]["dataGovernanceTagsInfo"] == {"dataGovernanceTags": {}}

    def test_remove_tag_when_explicit_empty_dict_or_none(self):
        fields = [
            {
                "name": "col_empty_dict",
                "type": "STRING",
                "dataGovernanceTagsInfo": {
                    "dataGovernanceTags": {"my-project/data_class": "email"}
                },
            },
            {
                "name": "col_none",
                "type": "STRING",
                "dataGovernanceTagsInfo": {
                    "dataGovernanceTags": {"my-project/data_class": "ssn"}
                },
            },
            {"name": "col_fresh_empty_dict", "type": "STRING"},
        ]
        columns = {
            "col_empty_dict": {"name": "col_empty_dict", "data_governance_tags": {}},
            "col_none": {"name": "col_none", "data_governance_tags": None},
            "col_fresh_empty_dict": {"name": "col_fresh_empty_dict", "data_governance_tags": {}},
        }

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["dataGovernanceTagsInfo"] == {"dataGovernanceTags": {}}
        assert wire[1]["dataGovernanceTagsInfo"] == {"dataGovernanceTags": {}}
        assert wire[2]["dataGovernanceTagsInfo"] == {"dataGovernanceTags": {}}

    def test_untouched_sibling_preserves_existing_governance_tag(self):
        fields = [
            {"name": "id", "type": "INTEGER"},
            {
                "name": "unlisted_secret",
                "type": "STRING",
                "dataGovernanceTagsInfo": {
                    "dataGovernanceTags": {"my-project/data_class": "ssn"}
                },
            },
        ]
        # Only `id` is documented in dbt; `unlisted_secret` must retain its existing tag.
        columns = {"id": {"name": "id", "description": "Primary key"}}

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert "dataGovernanceTagsInfo" not in wire[0]
        assert wire[1]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "ssn"}
        }

    def test_legacy_policy_tag_unaffected_without_dgt(self):
        policy_tag = "projects/p/locations/us/taxonomies/1/policyTags/2"
        fields = [{"name": "ssn", "type": "STRING"}]
        columns = {"ssn": {"name": "ssn", "policy_tags": [policy_tag]}}

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["policyTags"] == {"names": [policy_tag]}
        assert "dataGovernanceTagsInfo" not in wire[0]

    def test_atomic_swap_policy_tag_to_data_governance_tag(self):
        policy_tag = "projects/p/locations/us/taxonomies/1/policyTags/2"
        fields = [
            {
                "name": "ssn",
                "type": "STRING",
                "policyTags": {"names": [policy_tag]},
            }
        ]
        columns = {
            "ssn": {
                "name": "ssn",
                "data_governance_tags": {"my-project/data_class": "ssn"},
            }
        }

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        # Both empty policyTags and new dataGovernanceTagsInfo sent in a single patch.
        assert wire[0]["policyTags"] == {"names": []}
        assert wire[0]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "ssn"}
        }

    def test_atomic_rollback_data_governance_tag_to_policy_tag(self):
        policy_tag = "projects/p/locations/us/taxonomies/1/policyTags/2"
        fields = [
            {
                "name": "ssn",
                "type": "STRING",
                "dataGovernanceTagsInfo": {
                    "dataGovernanceTags": {"my-project/data_class": "ssn"}
                },
            }
        ]
        columns = {"ssn": {"name": "ssn", "policy_tags": [policy_tag]}}

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        # Both new policyTags and empty dataGovernanceTagsInfo sent in a single patch.
        assert wire[0]["policyTags"] == {"names": [policy_tag]}
        assert wire[0]["dataGovernanceTagsInfo"] == {"dataGovernanceTags": {}}

    def test_empty_policy_tags_allowed_with_data_governance_tags(self):
        fields = [{"name": "email", "type": "STRING"}]
        columns = {
            "email": {
                "name": "email",
                "policy_tags": [],
                "data_governance_tags": {"my-project/data_class": "email"},
            }
        }

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["policyTags"] == {"names": []}
        assert wire[0]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "email"}
        }

    def test_nested_struct_and_repeated_record_leaf_tags(self):
        fields = [
            {
                "name": "profile",
                "type": "RECORD",
                "fields": [
                    {"name": "work_email", "type": "STRING"},
                    {
                        "name": "identity",
                        "type": "RECORD",
                        "fields": [
                            {"name": "ssn", "type": "STRING"},
                            {
                                "name": "legacy_dgt",
                                "type": "STRING",
                                "dataGovernanceTagsInfo": {
                                    "dataGovernanceTags": {"my-project/data_class": "old"}
                                },
                            },
                        ],
                    },
                ],
            },
            {
                "name": "events",
                "type": "RECORD",
                "mode": "REPEATED",
                "fields": [
                    {"name": "actor_email", "type": "STRING"},
                    {"name": "kind", "type": "STRING"},
                ],
            },
        ]
        columns = {
            "profile": {"name": "profile", "description": "User profile struct"},
            "profile.work_email": {
                "name": "profile.work_email",
                "description": "Work email",
                "data_governance_tags": {"my-project/data_class": "email"},
            },
            "profile.identity.ssn": {
                "name": "profile.identity.ssn",
                "description": "US SSN",
                "data_governance_tags": {"my-project/data_class": "ssn"},
            },
            "profile.identity.legacy_dgt": {
                "name": "profile.identity.legacy_dgt",
                "description": "Cleared leaf",
            },
            "events.actor_email": {
                "name": "events.actor_email",
                "description": "Actor email",
                "data_governance_tags": {"my-project/domain": "security"},
            },
        }

        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])

        profile = wire[0]
        assert profile["description"] == "User profile struct"
        assert "dataGovernanceTagsInfo" not in profile
        work_email = profile["fields"][0]
        assert work_email["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "email"}
        }
        ssn = profile["fields"][1]["fields"][0]
        assert ssn["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "ssn"}
        }
        cleared = profile["fields"][1]["fields"][1]
        assert cleared["dataGovernanceTagsInfo"] == {"dataGovernanceTags": {}}

        events = wire[1]
        actor_email = events["fields"][0]
        assert actor_email["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/domain": "security"}
        }
        assert "dataGovernanceTagsInfo" not in events["fields"][1]

    def test_multiple_tags_mapping_passes_through_without_truncation(self):
        fields = [{"name": "email", "type": "STRING"}]
        columns = {
            "email": {
                "name": "email",
                "data_governance_tags": {
                    "my-project/data_class": "email",
                    "my-project/domain": "marketing",
                },
            }
        }
        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {
                "my-project/data_class": "email",
                "my-project/domain": "marketing",
            }
        }

    def test_error_on_both_policy_tags_and_data_governance_tags(self):
        fields = [{"name": "email", "type": "STRING"}]
        columns = {
            "email": {
                "name": "email",
                "policy_tags": ["projects/p/locations/us/taxonomies/1/policyTags/2"],
                "data_governance_tags": {"my-project/data_class": "email"},
            }
        }
        with pytest.raises(
            dbt_common.exceptions.CompilationError,
            match="cannot specify both `policy_tags` and `data_governance_tags`",
        ):
            self._run_update_columns(fields, columns)

    def test_error_on_record_or_struct_parent_column(self):
        for col_type in ("RECORD", "STRUCT"):
            fields = [
                {
                    "name": "profile",
                    "type": col_type,
                    "fields": [{"name": "email", "type": "STRING"}],
                }
            ]
            columns = {
                "profile": {
                    "name": "profile",
                    "data_governance_tags": {"my-project/data_class": "pii"},
                }
            }
            with pytest.raises(
                dbt_common.exceptions.CompilationError,
                match="RECORD/STRUCT column and cannot have `data_governance_tags`",
            ):
                self._run_update_columns(fields, columns)

    def test_error_on_non_table_relations_when_dgt_configured(self):
        for table_type, rel_type in (
            ("VIEW", RelationType.View),
            ("MATERIALIZED_VIEW", RelationType.MaterializedView),
            ("EXTERNAL", RelationType.External),
            ("SNAPSHOT", RelationType.Table),
        ):
            fields = [{"name": "email", "type": "STRING"}]
            columns = {
                "email": {
                    "name": "email",
                    "data_governance_tags": {"my-project/data_class": "email"},
                }
            }
            with pytest.raises(
                dbt_common.exceptions.CompilationError,
                match="only supports on table columns",
            ):
                self._run_update_columns(
                    fields, columns, table_type=table_type, relation_type=rel_type
                )

    def test_error_on_nested_leaf_of_view_when_dgt_configured(self):
        fields = [
            {
                "name": "profile",
                "type": "RECORD",
                "fields": [{"name": "email", "type": "STRING"}],
            }
        ]
        columns = {
            "profile.email": {
                "name": "profile.email",
                "data_governance_tags": {"my-project/data_class": "email"},
            }
        }
        with pytest.raises(
            dbt_common.exceptions.CompilationError,
            match="only supports on table columns",
        ):
            self._run_update_columns(
                fields, columns, table_type="VIEW", relation_type=RelationType.View
            )

    def test_view_with_description_only_succeeds_without_dgt_field(self):
        fields = [{"name": "email", "type": "STRING"}]
        columns = {"email": {"name": "email", "description": "View email description"}}
        client = self._run_update_columns(
            fields, columns, table_type="VIEW", relation_type=RelationType.View
        )
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["description"] == "View email description"
        assert "dataGovernanceTagsInfo" not in wire[0]

    def test_record_parent_with_empty_dgt_dict_allowed(self):
        fields = [
            {
                "name": "profile",
                "type": "RECORD",
                "fields": [{"name": "email", "type": "STRING"}],
            }
        ]
        columns = {"profile": {"name": "profile", "data_governance_tags": {}}}
        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert "dataGovernanceTagsInfo" not in wire[0]

    def test_empty_columns_short_circuits(self):
        mock_conn = MagicMock()
        relation = BigQueryRelation.create(
            database="dbt-unit-000000", schema="dummy_schema", identifier="customers"
        )
        adapter = self.get_adapter("oauth")
        adapter.connections.get_thread_connection = MagicMock(return_value=mock_conn)
        adapter.update_columns(relation, {})
        mock_conn.handle.get_table.assert_not_called()
        mock_conn.handle.update_table.assert_not_called()

    def test_extract_table_schema_dicts_fallback_and_non_mutation(self):
        field = SchemaField.from_api_repr(
            {
                "name": "email",
                "type": "STRING",
                "mode": "NULLABLE",
                "dataGovernanceTagsInfo": {
                    "dataGovernanceTags": {"my-project/data_class": "email"}
                },
            }
        )
        minimal_table = MagicMock(spec=["schema"])
        minimal_table.schema = [field]
        extracted = BigQueryAdapter._extract_table_schema_dicts(minimal_table)
        extracted[0]["description"] = "mutated"
        assert field.to_api_repr().get("description") is None
        assert extracted[0]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "email"}
        }

    def test_validation_rejects_invalid_shapes_and_values(self):
        invalid_cases = [
            (["my-project/key"], "expected a mapping"),
            ("my-project/key", "expected a mapping"),
            (123, "expected a mapping"),
            (True, "expected a mapping"),
            ({123: "val"}, "expected string"),
            ({"": "val"}, "expected '<parent_id>/<tag_key>'"),
            ({"no_slash": "val"}, "expected '<parent_id>/<tag_key>'"),
            ({"a/b/c": "val"}, "expected '<parent_id>/<tag_key>'"),
            ({"/only_key": "val"}, "expected '<parent_id>/<tag_key>'"),
            ({"project/": "val"}, "expected '<parent_id>/<tag_key>'"),
            ({"my project/key": "val"}, "expected '<parent_id>/<tag_key>'"),
            ({" my-project/key": "val"}, "expected '<parent_id>/<tag_key>'"),
            ({"my-project/key": ""}, "non-empty tag value short name"),
            ({"my-project/key": "   "}, "non-empty tag value short name"),
            ({"my-project/key": " email "}, "without leading or trailing whitespace"),
            ({"my-project/key": 42}, "non-empty tag value short name"),
            ({"my-project/key": None}, "non-empty tag value short name"),
            ({"my-project/key": "pii/email"}, "leaf tag value short name"),
        ]
        for raw_value, expected_msg in invalid_cases:
            with pytest.raises(dbt_common.exceptions.CompilationError, match=expected_msg):
                BigQueryAdapter._parse_data_governance_tags("email", raw_value)

    def test_mapping_valued_and_blank_policy_tags_allowed_with_data_governance_tags(self):
        fields = [
            {"name": "email", "type": "STRING"},
            {"name": "phone", "type": "STRING"},
        ]
        columns = {
            "email": {
                "name": "email",
                "policy_tags": [{"masking_policy": "mask_email"}],
                "data_governance_tags": {"my-project/data_class": "email"},
            },
            "phone": {
                "name": "phone",
                "policy_tags": ["", "   "],
                "data_governance_tags": {"my-project/data_class": "phone"},
            },
        }
        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["policyTags"] == {"names": []}
        assert wire[0]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "email"}
        }
        assert wire[1]["policyTags"] == {"names": []}
        assert wire[1]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "phone"}
        }

    def test_error_on_mixed_and_scalar_string_policy_tags_with_data_governance_tags(self):
        policy_tag = "projects/p/locations/us/taxonomies/1/policyTags/2"
        for raw_pt in ([policy_tag, {"masking_policy": "mask_email"}], policy_tag):
            fields = [{"name": "email", "type": "STRING"}]
            columns = {
                "email": {
                    "name": "email",
                    "policy_tags": raw_pt,
                    "data_governance_tags": {"my-project/data_class": "email"},
                }
            }
            with pytest.raises(
                dbt_common.exceptions.CompilationError,
                match="cannot specify both `policy_tags` and `data_governance_tags`",
            ):
                self._run_update_columns(fields, columns)

    def test_case_insensitive_unquoted_column_matching(self):
        fields = [
            {"name": "EMAIL", "type": "STRING"},
            {
                "name": "PROFILE",
                "type": "RECORD",
                "fields": [{"name": "WORK_EMAIL", "type": "STRING"}],
            },
            {"name": "QuotedMismatch", "type": "STRING"},
        ]
        columns = {
            "email": {
                "name": "email",
                "description": "Upper email",
                "data_governance_tags": {"my-project/data_class": "email"},
            },
            "profile.work_email": {
                "name": "profile.work_email",
                "data_governance_tags": {"my-project/data_class": "email"},
            },
            "quotedmismatch": {
                "name": "quotedmismatch",
                "quote": True,
                "data_governance_tags": {"my-project/data_class": "email"},
            },
        }
        client = self._run_update_columns(fields, columns)
        wire = _wire_fields(client.update_table.call_args.args[0])
        assert wire[0]["description"] == "Upper email"
        assert wire[0]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "email"}
        }
        assert wire[1]["fields"][0]["dataGovernanceTagsInfo"] == {
            "dataGovernanceTags": {"my-project/data_class": "email"}
        }
        assert "dataGovernanceTagsInfo" not in wire[2]

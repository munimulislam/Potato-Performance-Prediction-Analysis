with src as (
    SELECT *
    FROM {{ source('trial_data', 'quarantine') }}
),

staged_quarantine as (
    SELECT
        * EXCLUDE (_dlt_id, _dlt_load_id, source_file, source_row),
        _dlt_load_id               AS load_id,
        _dlt_id                    AS business_key,
        source_file                AS source_file_name,
        CAST(source_row AS BIGINT) AS source_row_number,
    FROM src
)

SELECT * FROM staged_quarantine
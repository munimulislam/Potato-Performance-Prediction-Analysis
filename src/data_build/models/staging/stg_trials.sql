WITH src AS (
    SELECT *
    FROM {{ source('trial_data', 'valid') }}
),

staged_valid AS (
    SELECT
        * EXCLUDE (_dlt_valid_from, _dlt_valid_to, _dlt_id, _dlt_load_id, source_file, source_row, plot, year),
        _dlt_valid_from            AS valid_from,
        _dlt_valid_to              AS valid_to,
        _dlt_load_id               AS load_id,
        _dlt_id                    AS business_key,
        source_file                AS source_file_name,
        CAST(source_row AS BIGINT) AS source_row_number,
        CAST(plot       AS BIGINT) AS plot,
        cast(year       AS BIGINT) AS year
    FROM src
)

SELECT * FROM staged_valid
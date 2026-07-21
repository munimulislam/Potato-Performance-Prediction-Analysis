WITH current_view AS (
    SELECT * FROM {{ ref('stg_trials') }}
    WHERE valid_to is NULL
),

loc_map AS (
    SELECT
        LOWER(TRIM(location)) as location_key,
        env_type
    FROM {{ ref('location_env_type') }}
),

enriched AS (
    SELECT
        cv.*,
        COALESCE (m.env_type, 'UNKNOWN') as env_type,

        CASE
            WHEN cv.experiment_name        IS NULL          THEN 'UNKNOWN'
            WHEN lower(cv.experiment_name) LIKE '%process%' THEN 'PROCESS'
            WHEN lower(cv.experiment_name) LIKE '%ware%'    THEN 'WARE'
            ELSE 'UNKNOWN'
        END AS trial_type,

        CASE
            WHEN cv.experiment_name        IS NULL       THEN 'UNKNOWN'
            WHEN lower(cv.experiment_name) LIKE '%clay%' THEN 'CLAY'
            WHEN lower(cv.experiment_name) LIKE '%sand%' THEN 'SAND'
            ELSE 'UNKNOWN'
        END AS soil_type,

        FROM current_view cv
        LEFT JOIN loc_map m ON LOWER(TRIM(cv.location)) = m.location_key
)

SELECT * FROM enriched
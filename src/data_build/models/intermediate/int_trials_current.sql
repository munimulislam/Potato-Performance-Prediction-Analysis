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
        cv.* EXCLUDE (o_a_score, appearance, tubersize, eveness, tubnumbers,
                      eyedepth, uniformity, ffscab, ffdefects, ffhollowh, ff_irs),

        COALESCE (m.env_type, 'UNKNOWN') AS env_type,

        CASE
            WHEN COALESCE(cv.finalyield, cv.yield_total_t_ha) < 0 
                OR COALESCE(cv.finalyield, cv.yield_total_t_ha) > 250 THEN NULL
            ELSE COALESCE(cv.finalyield, cv.yield_total_t_ha)
        END AS yield,

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

        CASE
            WHEN cv.o_a_score < 0 OR cv.o_a_score > 9 THEN NULL
            ELSE cv.o_a_score
        END AS o_a_score,

        CASE
            WHEN cv.appearance < 0 OR cv.appearance > 9 THEN NULL
            ELSE cv.appearance
        END AS appearance,

        CASE
            WHEN cv.tubersize < 0 OR cv.tubersize > 9 THEN NULL
            ELSE cv.tubersize
        END AS tubersize,
        
        CASE
            WHEN cv.eveness < 0 OR cv.eveness > 9 THEN NULL
            ELSE cv.eveness
        END AS eveness,

        CASE
            WHEN cv.tubnumbers < 0 OR cv.tubnumbers > 9 THEN NULL
            ELSE cv.tubnumbers
        END AS tubnumbers,

        CASE
            WHEN cv.eyedepth < 0 OR cv.eyedepth > 9 THEN NULL
            ELSE cv.eyedepth
        END AS eyedepth,

        CASE
            WHEN cv.uniformity < 0 OR cv.uniformity > 9 THEN NULL
            ELSE cv.uniformity
        END AS uniformity,

        CASE
            WHEN cv.ffscab < 0 OR cv.ffscab > 9 THEN NULL
            ELSE cv.ffscab
        END AS ffscab,

        CASE
            WHEN cv.ffdefects < 0 OR cv.ffdefects > 9 THEN NULL
            ELSE cv.ffdefects
        END AS ffdefects,

        CASE
            WHEN cv.ffhollowh < 0 OR cv.ffhollowh > 9 THEN NULL
            ELSE cv.ffhollowh
        END AS ffhollowh,

        CASE
            WHEN cv.ff_irs < 0 OR cv.ff_irs > 9 THEN NULL
            ELSE cv.ff_irs
        END AS ff_irs

        FROM current_view cv
        LEFT JOIN loc_map m ON LOWER(TRIM(cv.location)) = m.location_key
)

SELECT * FROM enriched
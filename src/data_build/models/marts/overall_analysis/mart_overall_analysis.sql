{{ config(materialized='table', schema='gold') }}

SELECT
    name1,
    year,
    location,
    experiment_name,
    plot,
    env_type,
    trial_type,
    soil_type,
    o_a_score,
    tubersize,
    eveness,
    appearance,
    tubnumbers,
    eyedepth,
    uniformity,
    ffscab,
    ffdefects,
    ffhollowh,
    ff_irs,
    yield
FROM {{ ref('int_trials_current') }}
WHERE o_a_score IS NOT NULL
AND env_type IN ('MED', 'NE')
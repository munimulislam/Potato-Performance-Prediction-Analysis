{{ config(materialized='table', schema='gold') }}

select
    name1,
    year,
    env_type,
    case when env_type = 'NE' then 1 else 0 end as env_enc,
    o_a_score,
    tubersize,
    eveness,
    appearance,
    tubnumbers,
    eyedepth,
    finalyield,
    uniformity,
    ffscab,
    ffdefects,
    ffhollowh,
    ff_irs
from {{ ref('int_trials_current') }}
where o_a_score is not null
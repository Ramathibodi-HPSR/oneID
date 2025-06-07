WITH UnifiedVisits AS (
    (
        SELECT
            t1.pid,
            t1.service_date,
            EXTRACT(YEAR FROM t1.service_date) AS p_visit_year,
            EXTRACT(MONTH FROM t1.service_date) AS p_visit_month,
            t1.prov1,
            t1.hcode,
            hm.nhso_zonename AS nhso_zone,
            t1.sub_type_name,
            hm.htype_name AS htype,
            'all_oppp_visits_wel_ucs' AS source_table
        FROM
            rama_one_id.all_oppp_visits_wel_ucs AS t1
        JOIN
            rama_one_id.v_l_hosmap AS hm
            ON t1.hcode = hm.hcode
        WHERE
            t1.pdx NOT IN ('U071', 'U119')   -- filter COVID
    )
    UNION ALL
    (
        SELECT
            t2.pid,
            t2.service_date,
            EXTRACT(YEAR FROM t2.service_date) AS p_visit_year,
            EXTRACT(MONTH FROM t2.service_date) AS p_visit_month,
            t2.province_name AS prov1,
            t2.hcode,
            hm.nhso_zonename AS nhso_zone,
            hm.sub_type_name,
            hm.htype_name AS htype,
            'v_t_nhso_one_id' AS source_table
        FROM
            rama_one_id.v_t_nhso_one_id AS t2
        JOIN
            rama_one_id.v_l_hosmap AS hm
            ON t2.hcode = hm.hcode
        WHERE
            hm.htype_name = 'เอกชน'
            AND t2.pdx NOT IN ('U071', 'U119')  
    )
),
PatientPreviousVisits AS (
    SELECT
        pid,
        service_date,
        p_visit_year AS visit_year,
        p_visit_month AS visit_month,
        prov1,
        hcode,
        nhso_zone,
        sub_type_name,
        htype,
        LAG(service_date) OVER (PARTITION BY pid ORDER BY service_date) AS previous_service_date
    FROM
        UnifiedVisits
),
NewcomerFlaggedVisits AS (
    SELECT
        pid,
        service_date,
        visit_year,
        visit_month,
        prov1,
        hcode,
        nhso_zone,
        sub_type_name,
        htype,
        CASE
            WHEN previous_service_date IS NULL THEN 1
            WHEN service_date >= ADD_MONTHS(previous_service_date, 36) THEN 1
            ELSE 0
        END AS is_newcomer
    FROM
        PatientPreviousVisits
)
SELECT
    visit_year,
    visit_month,
    prov1,
    nhso_zone,
    sub_type_name,
    htype,
    COUNT(DISTINCT pid) AS newcomers_count
FROM
    NewcomerFlaggedVisits
WHERE
    is_newcomer = 1
    AND visit_year >= 2022
GROUP BY
    visit_year,
    visit_month,
    prov1,
    nhso_zone,
    sub_type_name,
    htype
ORDER BY
    visit_year,
    visit_month,
    prov1,
    nhso_zone,
    sub_type_name,
    htype;

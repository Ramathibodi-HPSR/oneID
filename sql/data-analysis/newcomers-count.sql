-- Step 2: For each patient and their visit, find the immediately preceding visit date.
WITH PatientPreviousVisits AS (
    SELECT
        visits.pid,
        visits.service_date,
        visits.p_visit_year AS visit_year,
        visits.p_visit_month AS visit_month,
        visits.prov1,
        visits.hcode,
        hosmap.nhso_zonename AS nhso_zone,
        visits.sub_type_name, -- ADDED: Select sub_type_name from the visits table
        LAG(visits.service_date) OVER (PARTITION BY visits.pid ORDER BY visits.service_date) AS previous_service_date
    FROM
        rama_one_id.all_oppp_visits_wel_ucs visits
    LEFT JOIN
        rama_one_id.v_l_hosmap hosmap
        ON visits.hcode = hosmap.hcode
)
-- Step 3: Flag each visit as a "newcomer" visit based on the 3-year (month/date precise) rule.
,NewcomerFlaggedVisits AS (
    SELECT
        pid,
        service_date,
        visit_year,
        visit_month,
        prov1,
        hcode,
        nhso_zone,
        sub_type_name, -- ADDED: Pass sub_type_name through
        CASE
            WHEN previous_service_date IS NULL THEN 1
            WHEN service_date >= ADD_MONTHS(previous_service_date, 36) THEN 1
            ELSE 0
        END AS is_newcomer
    FROM
        PatientPreviousVisits
)
-- Step 4: Count the distinct newcomers for each month, year, prov1, nhso_zone, and sub_type_name, starting from 2022.
SELECT
    visit_year,
    visit_month,
    prov1,
    nhso_zone,
    sub_type_name,
    COUNT(DISTINCT pid) AS newcomers_count
FROM
    NewcomerFlaggedVisits
WHERE
    is_newcomer = 1
    AND visit_year >= 2022
GROUP BY
    -- If you do not want any following variables, comment that line.
    visit_year,
    visit_month,
    prov1,
    nhso_zone,
    sub_type_name
ORDER BY
    -- If you do not want any following variables, comment that line.
    visit_month,
    prov1,
    nhso_zone,
    sub_type_name;

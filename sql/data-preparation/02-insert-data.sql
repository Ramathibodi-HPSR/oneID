INSERT OVERWRITE TABLE rama_one_id.all_oppp_visits_wel_ucs PARTITION (p_visit_year, p_visit_month)
SELECT
    visits.pid,
    visits.service_date,
    visits.hcode,
    visits.hmain,
    visits.maininscl,
    visits.prov1,
    visits.pdx,
    hosmap.sub_type_name,
    hosmap.nhso_zonename
FROM
    rama_one_id.all_oppp_visits_processed visits
LEFT JOIN
    rama_one_id.v_l_hosmap hosmap
    ON visits.hcode = hosmap.hcode
WHERE
    visits.maininscl IN ('WEL', 'UCS');

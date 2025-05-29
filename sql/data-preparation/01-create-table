-- Drop the existing table
DROP TABLE IF EXISTS rama_one_id.all_oppp_visits_wel_ucs;

-- Create the new table combines all relevant OPD tables
CREATE TABLE rama_one_id.all_oppp_visits_wel_ucs (
    pid STRING,
    service_date DATE,
    hcode STRING,
    hmain STRING,
    maininscl STRING,
    prov1 STRING,
    pdx STRING,
    sub_type_name STRING,
    nhso_zonename STRING
)
PARTITIONED BY (p_visit_year INT, p_visit_month INT)
STORED AS PARQUET
TBLPROPERTIES ('parquet.compress'='SNAPPY');

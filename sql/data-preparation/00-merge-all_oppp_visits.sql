-- Create the new consolidated table
-- Adjust table name, column types, and partitioning strategy as per your system's best practices.
CREATE TABLE rama_one_id.all_oppp_visits_processed (
    pid STRING, -- Assuming pid is a string, adjust if it's numeric (e.g., BIGINT)
    service_date DATE,
    visit_year INT,
    visit_month INT,
    -- Add other columns you might need for future analysis (e.g., hcode, hmain, maininscl, prov1, pdx)
    hcode STRING,
    hmain STRING,
    maininscl STRING,
    prov1 STRING,
    pdx STRING
)
-- Example for Hive/Spark partitioning:
PARTITIONED BY (p_visit_year INT, p_visit_month INT) -- Partition by year and month for efficient filtering
STORED AS PARQUET -- Or ORC, etc.
TBLPROPERTIES ('parquet.compress'='SNAPPY'); -- Example compression

-- Insert data into the new table
INSERT OVERWRITE TABLE rama_one_id.all_oppp_visits_processed PARTITION (p_visit_year, p_visit_month)
SELECT
    pid,
    CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE) AS service_date,
    EXTRACT(YEAR FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)) AS visit_year,
    EXTRACT(MONTH FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)) AS visit_month,
    hcode, hmain, maininscl, prov1, pdx
FROM rama_one_id.v_t_f_oppp_indiv_flat_y62
UNION ALL
SELECT
    pid,
    CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE),
    EXTRACT(YEAR FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    EXTRACT(MONTH FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    hcode, hmain, maininscl, prov1, pdx
FROM rama_one_id.v_t_f_oppp_indiv_flat_y63
UNION ALL
SELECT
    pid,
    CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE),
    EXTRACT(YEAR FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    EXTRACT(MONTH FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    hcode, hmain, maininscl, prov1, pdx
FROM rama_one_id.v_t_f_oppp_indiv_flat_y64
UNION ALL
SELECT
    pid,
    CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE),
    EXTRACT(YEAR FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    EXTRACT(MONTH FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    hcode, hmain, maininscl, prov1, pdx
FROM rama_one_id.v_t_f_oppp_indiv_flat_y65
UNION ALL
SELECT
    pid,
    CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE),
    EXTRACT(YEAR FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    EXTRACT(MONTH FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    hcode, hmain, maininscl, prov1, pdx
FROM rama_one_id.v_t_f_oppp_indiv_flat_y66
UNION ALL
SELECT
    pid,
    CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE),
    EXTRACT(YEAR FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    EXTRACT(MONTH FROM CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(date_serv, 'yyyyMMdd')) AS DATE)),
    hcode, hmain, maininscl, prov1, pdx
FROM rama_one_id.v_t_f_opd_indiv_flat_y67;

-- After insertion, run ANALYZE TABLE to collect statistics (important for optimizer)
ANALYZE TABLE rama_one_id.all_oppp_visits_processed COMPUTE STATISTICS FOR COLUMNS pid, service_date, p_visit_year, p_visit_month;

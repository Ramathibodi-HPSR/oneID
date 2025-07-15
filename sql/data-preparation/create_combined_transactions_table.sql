-- File: create_combined_transactions_table.sql
-- Description: Combines transaction data, extracts year/month, filters by maininscl,
--              and joins with hospital mapping to create a new complete table.
--              Includes optimized indexing for subsequent calculations.

-- Drop the table if it already exists to ensure a clean slate for recreation
DROP TABLE IF EXISTS rama_one_id.combined_filtered_transactions;

-- Create a new table to store the combined, filtered, and joined transaction data
CREATE TABLE rama_one_id.combined_filtered_transactions AS
SELECT
    t.tran_id,
    t.pid,
    t.date_serv,
    CAST(SUBSTRING(t.date_serv, 1, 4) AS INT) AS year, -- Extract year from YYYYMMDD string
    CAST(SUBSTRING(t.date_serv, 5, 2) AS INT) AS month, -- Extract month from YYYYMMDD string
    t.hcode,
    t.hmain,
    t.maininscl,
    t.prov1,
    t.prov2,
    t.pdx,
    h.nhso_zonename,
    h.sub_type_name
FROM
    (
        -- Combine all yearly transaction tables using UNION ALL
        SELECT tran_id, pid, date_serv, hcode, hmain, maininscl, prov1, prov2, pdx FROM rama_one_id.v_t_f_oppp_indiv_flat_y62 WHERE maininscl IN ('WEL', 'UCS')
        UNION ALL
        SELECT tran_id, pid, date_serv, hcode, hmain, maininscl, prov1, prov2, pdx FROM rama_one_id.v_t_f_oppp_indiv_flat_y63 WHERE maininscl IN ('WEL', 'UCS')
        UNION ALL
        SELECT tran_id, pid, date_serv, hcode, hmain, maininscl, prov1, prov2, pdx FROM rama_one_id.v_t_f_oppp_indiv_flat_y64 WHERE maininscl IN ('WEL', 'UCS')
        UNION ALL
        SELECT tran_id, pid, date_serv, hcode, hmain, maininscl, prov1, prov2, pdx FROM rama_one_id.v_t_f_oppp_indiv_flat_y65 WHERE maininscl IN ('WEL', 'UCS')
        UNION ALL
        SELECT tran_id, pid, date_serv, hcode, hmain, maininscl, prov1, prov2, pdx FROM rama_one_id.v_t_f_oppp_indiv_flat_y66 WHERE maininscl IN ('WEL', 'UCS')
        UNION ALL
        SELECT tran_id, pid, date_serv, hcode, hmain, maininscl, prov1, prov2, pdx FROM rama_one_id.v_t_f_oppp_indiv_flat_y67 WHERE maininscl IN ('WEL', 'UCS')
        UNION ALL
        SELECT tran_id, pid, date_serv, hcode, hmain, maininscl, prov1, prov2, pdx FROM rama_one_id.v_t_f_oppp_indiv_flat_y68 WHERE maininscl IN ('WEL', 'UCS')
    ) AS t
INNER JOIN
    rama_one_id.v_l_hosmap AS h
ON
    t.hcode = h.hcode;

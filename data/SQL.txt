INSERT INTO s_grnplm_vd_t_bvd_db_dmslcl.d_agr_cred  SELECT agr.agr_cred_id,
    agr.prnt_agr_cred_id,
    agr.info_system_id,
    agr.agr_num,
    agr.grntee_agr_num,
    agr.signed_dt,
    agr.issue_crncy_id,
    agr.eks_overdraft_flag,
    agr.agr_regls_type_id,
        CASE
            WHEN (agr.eks_issue_dt IS NOT NULL) THEN agr.eks_issue_dt
            WHEN ((agr.agr_cred_type_cd = '1'::text) AND (agr.eks_transhes_issue_dt IS NOT NULL)) THEN agr.eks_transhes_issue_dt
            ELSE aco_fst_iss.optn_issue_dt
        END AS issue_dt,
    agr.close_dt,
    agr.expiration_dt,
    agr.host_agr_cred_id,
    agr.host_crm_id,
    agr.eks_agr_cred_type_id,
    agr.agr_frame_id,
    agr.eks_rvlng_flag,
    agr.issue_int_org_id,
    agr.multi_crncy_flag,
    agr.eks_old_restruct_cnt,
    agr.eks_old_prolong_cnt,
    agr.eks_contract_expiration_dt,
    NULL::text AS diff_intst_rate_flag,
    agr.grntee_claim_notice_dt,
    agr.restruct_refin_default_flag,
    agr.restruct_refin_default_dt,
    coa_crncy.agra_l001_crncy_id AS debt_due_acc_crncy_id,
    coa_crncy.agra_l007_crncy_id AS unused_limit_acc_crncy_id,
    coa_crncy.agra_l009_crncy_id AS grntee_acc_crncy_id,
    agr.has_higher_flag,
    agr.zero_id,
    agr.four_id,
    agr.eks_cession_dt,
    agr.agr_cession_id,
    agr.cession_sell_got_cess_amt,
    agr.cession_sell_debt_cess_amt,
    agr.cession_sell_intrst_cess_amt,
    agr.cession_sell_cess_amt,
        CASE
            WHEN (agr.f26_agr_cred_type_cd = 'Г'::text) THEN COALESCE(coa_crncy.agra_l009_crncy_id, agr.issue_crncy_id)
            WHEN (agr.agr_cred_type_cd = '0'::text) THEN COALESCE(coa_crncy.agra_l001_crncy_id, agr.issue_crncy_id)
            WHEN (agr.agr_cred_type_cd = '4'::text) THEN COALESCE(coa_crncy.agra_l001_crncy_id, agr.issue_crncy_id)
            WHEN (agr.agr_cred_type_cd = '1'::text) THEN COALESCE(coa_crncy.agra_l007_crncy_id, agr.issue_crncy_id)
            ELSE agr.issue_crncy_id
        END AS crncy_id,
    agr.agr_cred_type_id,
    agr.agr_cred_type_cd,
    agr.f26_agr_cred_type_id,
    agr.f26_agr_cred_type_cd,
    agr.rvlng_flag,
    agr.overdraft_flag,
        CASE
            WHEN (cess_buy_coa.agr_cred_id IS NOT NULL) THEN cess_buy_coa.cess_buy_coa_open_dt
            ELSE COALESCE(agr.eks_cession_dt,
            CASE
                WHEN (cess_sell_debt_coa.cess_debt_coa_open_dt >= COALESCE(cess_sell_debt_coa.coa_bind_dt, '1900-01-01'::date)) THEN cess_sell_debt_coa.cess_debt_coa_open_dt
                ELSE cess_sell_debt_coa.coa_bind_dt
            END, agr.cess_last_reg_dt)
        END AS cession_dt,
        CASE
            WHEN (cess_buy_coa.agr_cred_id IS NOT NULL) THEN 'Y'::text
            ELSE 'N'::text
        END AS cession_buy_flag,
        CASE
            WHEN (((cess_sell_debt_coa.agr_cred_id IS NOT NULL) OR (agr.cess_last_reg_dt IS NOT NULL)) AND (cess_buy_coa.agr_cred_id IS NULL)) THEN 'Y'::text
            ELSE 'N'::text
        END AS cess_deferred_flag,
    rat_first.agr_cred_rate_first_dt,
    agr.cess_crncy_id,
    agr.k7m_flag,
    agr.poci_flag,
    agr.fin_guarantee_flag,
    agr.fin_guarantee_loan_flag,
    agr.limit_use_end_dt,
    agr.open_dt,
    agr.eks_grntee_amt,
    agr.grntee_ast_flag,
    NULL::bigint AS crncy_id_uvdo,
    NULL::text AS crncy_cd_uvdo,
    NULL::text AS iso_crncy_cd_uvdo,
    NULL::bigint AS cust_id_uvdo,
    NULL::bigint AS letter_of_credit_type_id,
    NULL::text AS let_of_cred_revoce_flag,
    NULL::text AS let_of_cred_cover_flag,
    NULL::bigint AS host_let_of_cred_cover_id,
    NULL::numeric(38,8) AS agr_cred_start_amt,
    agr.k7m_pl_flag,
    NULL::bigint AS coa_cred_f26_prod_id,
    cst.subject_area_type_id,
    NULL::bigint AS let_of_cred_acc_scheme_id
   FROM ((((((s_grnplm_vd_t_bvd_db_dmslcl.d_agr_cred_tmp agr
     LEFT JOIN ( SELECT eatc.agrmnt_id AS agr_cred_id,
            min(eatc.agrmnt_gl_acct_start_dt) AS coa_bind_dt,
            min(c.coa_start_dt) AS cess_debt_coa_open_dt
           FROM ((s_grnplm_as_t_didsd_010_vd_dwh."v_$eks_agrmnt_to_coa_3" eatc
             LEFT JOIN s_grnplm_as_t_didsd_010_vd_dwh.v_coa c ON ((c.coa_id = eatc.coa_id)))
             LEFT JOIN s_grnplm_as_t_didsd_010_vd_dwh.v_gl_main_acct glma ON ((glma.gl_main_acct_id = c.gl_main_acct_id)))
          WHERE (substr(glma.gl_main_acct_num, 1, 7) = '4742325'::text)
          GROUP BY eatc.agrmnt_id) cess_sell_debt_coa ON ((cess_sell_debt_coa.agr_cred_id = agr.agr_cred_id)))
     LEFT JOIN ( SELECT c.agr_cred_id,
            min(c.start_dt) AS cess_buy_coa_open_dt
           FROM s_grnplm_vd_t_bvd_db_dmslcl.a_agr_cred_coa_period c
          WHERE ((((1 = 1) AND ((c.coa_num ~~ '478%'::text) OR (substr(c.coa_num, 1, 5) = ANY (ARRAY['44111'::text, '44211'::text, '44311'::text, '44411'::text, '44511'::text, '44611'::text, '44711'::text, '44811'::text, '44911'::text, '45011'::text, '45111'::text, '45211'::text, '45311'::text, '45411'::text, '45611'::text])))) AND (substr(c.meas_cd, 1, 6) = 'AGRA_L'::text)) AND (c.meas_rub > (0)::numeric))
          GROUP BY c.agr_cred_id) cess_buy_coa ON ((cess_buy_coa.agr_cred_id = agr.agr_cred_id)))
     LEFT JOIN ( SELECT coa_crncs_sub.agr_cred_id,
            max(
                CASE
                    WHEN (coa_crncs_sub.meas_cd = 'AGRA_L001'::text) THEN coa_crncs_sub.crncy_id
                    ELSE NULL::bigint
                END) AS agra_l001_crncy_id,
            max(
                CASE
                    WHEN (coa_crncs_sub.meas_cd = 'AGRA_L007'::text) THEN coa_crncs_sub.crncy_id
                    ELSE NULL::bigint
                END) AS agra_l007_crncy_id,
            max(
                CASE
                    WHEN (coa_crncs_sub.meas_cd = 'AGRA_L009'::text) THEN coa_crncs_sub.crncy_id
                    ELSE NULL::bigint
                END) AS agra_l009_crncy_id
           FROM ( SELECT t.agr_cred_id,
                    t.meas_cd,
                    t.crncy_id
                   FROM ( SELECT a_agr_cred_coa_period.agr_cred_id,
                            a_agr_cred_coa_period.meas_cd,
                            a_agr_cred_coa_period.crncy_id,
                            row_number() OVER (PARTITION BY a_agr_cred_coa_period.agr_cred_id, a_agr_cred_coa_period.meas_cd ORDER BY NULL::text) AS rn,
                            min(a_agr_cred_coa_period.crncy_id) OVER (PARTITION BY a_agr_cred_coa_period.agr_cred_id, a_agr_cred_coa_period.meas_cd) AS mn,
                            max(a_agr_cred_coa_period.crncy_id) OVER (PARTITION BY a_agr_cred_coa_period.agr_cred_id, a_agr_cred_coa_period.meas_cd) AS mx
                           FROM s_grnplm_vd_t_bvd_db_dmslcl.a_agr_cred_coa_period
                          WHERE (a_agr_cred_coa_period.meas_cd = ANY (ARRAY['AGRA_L001'::text, 'AGRA_L007'::text, 'AGRA_L009'::text]))) t
                  WHERE ((t.rn = 1) AND (t.mx = t.mn))) coa_crncs_sub
          GROUP BY coa_crncs_sub.agr_cred_id) coa_crncy ON ((coa_crncy.agr_cred_id = agr.agr_cred_id)))
     LEFT JOIN ( SELECT t.agr_cred_id,
            t.optn_issue_dt
           FROM ( SELECT aco_fst_iss_1.agr_cred_id,
                    aco_fst_iss_1.optn_dt AS optn_issue_dt,
                    row_number() OVER (PARTITION BY aco_fst_iss_1.agr_cred_id ORDER BY aco_fst_iss_1.optn_dt) AS rn
                   FROM s_grnplm_vd_t_bvd_db_dmslcl.d_agr_cred_optn aco_fst_iss_1
                  WHERE (((aco_fst_iss_1.issue_flag = 'Y'::text) OR (aco_fst_iss_1.move_flag = 'Y'::text)) AND (aco_fst_iss_1.optn_rub <> (0)::numeric))) t
          WHERE (t.rn = 1)) aco_fst_iss ON ((aco_fst_iss.agr_cred_id = agr.agr_cred_id)))
     LEFT JOIN ( SELECT rat.agrmnt_id AS agr_cred_id,
            min(rat.loan_agrmnt_rate_start_dt) AS agr_cred_rate_first_dt
           FROM s_grnplm_as_t_didsd_010_vd_dwh.v_loan_agrmnt_rate rat
          WHERE ((rat.rate_val > (0)::numeric) AND (rat.rate_calc_rule_id = ANY (ARRAY[37004, 13104, 45804, 67904, 35204])))
          GROUP BY rat.agrmnt_id) rat_first ON ((rat_first.agr_cred_id = agr.agr_cred_id)))
     LEFT JOIN ( SELECT v_cust.agr_cred_id,
            max(
                CASE
                    WHEN ((COALESCE(v_cust.ind_flag, 'N'::text) = 'Y'::text) AND (COALESCE(v_cust.org_flag, 'N'::text) = 'N'::text)) THEN 2
                    WHEN (COALESCE(v_cust.bank_flag, 'N'::text) = 'Y'::text) THEN 3
                    ELSE 1
                END) AS subject_area_type_id
           FROM s_grnplm_vd_t_bvd_db_dmslcl.d_agr_cred_cust v_cust
          WHERE ((('now'::text)::date >= v_cust.start_dt) AND (('now'::text)::date <= v_cust.end_dt))
          GROUP BY v_cust.agr_cred_id) cst ON ((agr.agr_cred_id = cst.agr_cred_id)))
UNION ALL
 SELECT agr_cred.agr_cred_id,
    NULL::bigint AS prnt_agr_cred_id,
    agr_cred.info_system_id,
    agr_cred.agr_num,
    NULL::text AS grntee_agr_num,
    agr_cred.open_dt AS signed_dt,
    NULL::bigint AS issue_crncy_id,
    NULL::text AS eks_overdraft_flag,
    agr_cred.uvdo_accred_scheme_id AS agr_regls_type_id,
    NULL::date AS issue_dt,
    agr_cred.close_dt,
    agr_cred.expiration_dt,
    (agr_cred.host_agr_cred_id)::text AS host_agr_cred_id,
    NULL::text AS host_crm_id,
    NULL::bigint AS eks_agr_cred_type_id,
        CASE
            WHEN (flt.agr_cred_type_id = '-1034'::integer) THEN agr_cred.agr_cred_id
            ELSE gen.agr_frame_id
        END AS agr_frame_id,
    NULL::text AS eks_rvlng_flag,
    NULL::bigint AS issue_int_org_id,
    NULL::text AS multi_crncy_flag,
    NULL::smallint AS eks_old_restruct_cnt,
    NULL::smallint AS eks_old_prolong_cnt,
    NULL::date AS eks_contract_expiration_dt,
    NULL::text AS diff_intst_rate_flag,
    NULL::date AS grntee_claim_notice_dt,
    NULL::text AS restruct_refin_default_flag,
    NULL::date AS restruct_refin_default_dt,
    NULL::bigint AS debt_due_acc_crncy_id,
    NULL::bigint AS unused_limit_acc_crncy_id,
    NULL::bigint AS grntee_acc_crncy_id,
    'N'::text AS has_higher_flag,
    NULL::bigint AS zero_id,
    NULL::bigint AS four_id,
    NULL::date AS eks_cession_dt,
    NULL::bigint AS agr_cession_id,
    NULL::numeric(38,8) AS cession_sell_got_cess_amt,
    NULL::numeric(38,8) AS cession_sell_debt_cess_amt,
    NULL::numeric(38,8) AS cession_sell_intrst_cess_amt,
    NULL::numeric(38,8) AS cession_sell_cess_amt,
    agr_crncy.crncy_id_eks AS crncy_id,
    flt.agr_cred_type_id,
    NULL::text AS agr_cred_type_cd,
    NULL::bigint AS f26_agr_cred_type_id,
    NULL::text AS f26_agr_cred_type_cd,
    rvlng.rvlng_flag,
    NULL::text AS overdraft_flag,
    NULL::date AS cession_dt,
    NULL::text AS cession_buy_flag,
    NULL::text AS cess_deferred_flag,
    NULL::date AS agr_cred_rate_first_dt,
    NULL::bigint AS cess_crncy_id,
    NULL::text AS k7m_flag,
    NULL::text AS poci_flag,
    NULL::text AS fin_guarantee_flag,
    NULL::text AS fin_guarantee_loan_flag,
    NULL::date AS limit_use_end_dt,
    agr_cred.open_dt,
    NULL::numeric(38,8) AS eks_grntee_amt,
    NULL::text AS grntee_ast_flag,
    agr_crncy.crncy_id AS crncy_id_uvdo,
    agr_crncy.crncy_cd AS crncy_cd_uvdo,
    agr_crncy.iso_crncy_cd AS iso_crncy_cd_uvdo,
    agr_cred.payer_cust_id AS cust_id_uvdo,
    agr_cred.agr_cred_type_id AS letter_of_credit_type_id,
    agr_cred.revocable_accred_flag AS let_of_cred_revoce_flag,
        CASE
            WHEN ((gen.agr_frame_id IS NOT NULL) OR (flt.agr_cred_type_id = '-1034'::integer)) THEN 'N'::text
            ELSE 'Y'::text
        END AS let_of_cred_cover_flag,
    agr_cred.host_eks_agrmnt_id AS host_let_of_cred_cover_id,
    agr_cred.start_sum_crncy_amt AS agr_cred_start_amt,
    NULL::text AS k7m_pl_flag,
    NULL::bigint AS coa_cred_f26_prod_id,
    (1)::smallint AS subject_area_type_id,
    rvlng.accounting_scheme_id AS let_of_cred_acc_scheme_id
   FROM ((((s_grnplm_as_t_didsd_029_vd_dwh.v_agr_cred agr_cred
     JOIN s_grnplm_vd_t_bvd_db_dmslcl.d_agr_cred_core_uvdo flt ON ((agr_cred.agr_cred_id = flt.agr_cred_id)))
     LEFT JOIN ( SELECT curu.crncy_id,
            max(cure.crncy_id) AS crncy_id_eks,
            max(COALESCE(cure.crncy_cd, '-1'::text)) AS crncy_cd,
            max(COALESCE(cure.crncy_name, ''::text)) AS crncy_name,
            max(curu.iso_crncy_cd) AS iso_crncy_cd,
            max(curu.info_system_id) AS info_system_id
           FROM (s_grnplm_as_t_didsd_029_vd_dwh.v_crncy curu
             LEFT JOIN s_grnplm_as_t_didsd_010_vd_dwh.v_crncy cure ON ((cure.iso_crncy_cd = curu.iso_crncy_cd)))
          GROUP BY curu.crncy_id) agr_crncy ON ((agr_crncy.crncy_id = agr_cred.crncy_id)))
     LEFT JOIN ( SELECT t1.agr_cred_id,
            max(t2.agr_cred_id) AS agr_frame_id
           FROM (( SELECT "v_agr_cred_metric_hist$$$".agr_cred_id,
                    max("v_agr_cred_metric_hist$$$".agr_cred_metric_txt) AS frame_host_agr_cred_id
                   FROM s_grnplm_as_t_didsd_029_vd_dwh."v_agr_cred_metric_hist$$$"
                  WHERE (((("v_agr_cred_metric_hist$$$".agr_cred_metric_type_id = '-1007'::integer) AND ("v_agr_cred_metric_hist$$$".agr_cred_metric_txt <> '-1'::text)) AND (btrim("v_agr_cred_metric_hist$$$".agr_cred_metric_txt) >= '0'::text)) AND (btrim("v_agr_cred_metric_hist$$$".agr_cred_metric_txt) <= '99999999999999999999'::text))
                  GROUP BY "v_agr_cred_metric_hist$$$".agr_cred_id) t1
             JOIN ( SELECT d_agr_cred_core_uvdo.agr_cred_id,
                    d_agr_cred_core_uvdo.host_agr_cred_id
                   FROM s_grnplm_vd_t_bvd_db_dmslcl.d_agr_cred_core_uvdo
                  WHERE (d_agr_cred_core_uvdo.agr_cred_type_id = '-1034'::integer)) t2 ON ((t1.frame_host_agr_cred_id = t2.host_agr_cred_id)))
          GROUP BY t1.agr_cred_id) gen ON ((agr_cred.agr_cred_id = gen.agr_cred_id)))
     LEFT JOIN ( SELECT agr.agr_cred_id,
            agr.accounting_scheme_id,
                CASE
                    WHEN ((aty.host_agr_cred_type_id = (12)::numeric) AND (ttt.uvdo_ttovarlc_name = 'Прочие товары'::text)) THEN 'Y'::text
                    ELSE 'N'::text
                END AS rvlng_flag
           FROM ((s_grnplm_as_t_didsd_029_vd_dwh.v_agr_cred agr
             LEFT JOIN s_grnplm_as_t_didsd_029_vd_dwh."v_$uvdo_ttovarlc" ttt ON ((ttt.uvdo_ttovarlc_id = agr.uvdo_ttovarlc_id)))
             LEFT JOIN s_grnplm_as_t_didsd_029_vd_dwh.v_agr_cred_type aty ON ((aty.agr_cred_type_id = agr.agr_cred_type_id)))) rvlng ON ((agr_cred.agr_cred_id = rvlng.agr_cred_id)))
UNION ALL
 SELECT core.agr_cred_id,
    NULL::bigint AS prnt_agr_cred_id,
    core.info_system_id,
    coa.coa_num AS agr_num,
    NULL::text AS grntee_agr_num,
    coa.coa_start_dt AS signed_dt,
    coa.crncy_id AS issue_crncy_id,
    NULL::text AS eks_overdraft_flag,
    NULL::bigint AS agr_regls_type_id,
    NULL::date AS issue_dt,
    coa.coa_end_dt AS close_dt,
    NULL::date AS expiration_dt,
    '-1'::text AS host_agr_cred_id,
    NULL::text AS host_crm_id,
    NULL::bigint AS eks_agr_cred_type_id,
    NULL::bigint AS agr_frame_id,
    NULL::text AS eks_rvlng_flag,
    xref.coa_int_org_pty_id AS issue_int_org_id,
    NULL::text AS multi_crncy_flag,
    NULL::smallint AS eks_old_restruct_cnt,
    NULL::smallint AS eks_old_prolong_cnt,
    NULL::date AS eks_contract_expiration_dt,
    NULL::text AS diff_intst_rate_flag,
    NULL::date AS grntee_claim_notice_dt,
    NULL::text AS restruct_refin_default_flag,
    NULL::date AS restruct_refin_default_dt,
    NULL::bigint AS debt_due_acc_crncy_id,
    NULL::bigint AS unused_limit_acc_crncy_id,
    NULL::bigint AS grntee_acc_crncy_id,
    NULL::text AS has_higher_flag,
    NULL::bigint AS zero_id,
    NULL::bigint AS four_id,
    NULL::date AS eks_cession_dt,
    NULL::bigint AS agr_cession_id,
    NULL::numeric(38,8) AS cession_sell_got_cess_amt,
    NULL::numeric(38,8) AS cession_sell_debt_cess_amt,
    NULL::numeric(38,8) AS cession_sell_intrst_cess_amt,
    NULL::numeric(38,8) AS cession_sell_cess_amt,
    coa.crncy_id,
    core.agr_cred_type_id,
    NULL::text AS agr_cred_type_cd,
    (4)::bigint AS f26_agr_cred_type_id,
    '1'::text AS f26_agr_cred_type_cd,
    NULL::text AS rvlng_flag,
    NULL::text AS overdraft_flag,
    NULL::date AS cession_dt,
    NULL::text AS cession_buy_flag,
    NULL::text AS cess_deferred_flag,
    NULL::date AS agr_cred_rate_first_dt,
    NULL::bigint AS cess_crncy_id,
    NULL::text AS k7m_flag,
    NULL::text AS poci_flag,
    NULL::text AS fin_guarantee_flag,
    NULL::text AS fin_guarantee_loan_flag,
    NULL::date AS limit_use_end_dt,
    coa.coa_start_dt AS open_dt,
    NULL::numeric(38,8) AS eks_grntee_amt,
    NULL::text AS grntee_ast_flag,
    NULL::bigint AS crncy_id_uvdo,
    NULL::text AS crncy_cd_uvdo,
    NULL::text AS iso_crncy_cd_uvdo,
    NULL::bigint AS cust_id_uvdo,
    NULL::bigint AS letter_of_credit_type_id,
    NULL::text AS let_of_cred_revoce_flag,
    NULL::text AS let_of_cred_cover_flag,
    NULL::bigint AS host_let_of_cred_cover_id,
    NULL::numeric(38,8) AS agr_cred_start_amt,
    NULL::text AS k7m_pl_flag,
    type1.agrmnt_class_val_id AS coa_cred_f26_prod_id,
    core.subject_area_type_id,
    NULL::bigint AS let_of_cred_acc_scheme_id
   FROM ((((s_grnplm_vd_t_bvd_db_dmslcl.d_coa_cred_core core
     JOIN s_grnplm_as_t_didsd_010_vd_dwh.v_coa coa ON ((core.coa_id = coa.coa_id)))
     LEFT JOIN s_grnplm_as_t_didsd_010_vd_dwh.v_coa_int_org_xref xref ON (((((coa.coa_id = xref.coa_id) AND (xref.coa_int_org_type_id = '-1001'::integer)) AND (coa.coa_start_dt >= xref.coa_int_org_start_dt)) AND (coa.coa_start_dt <= xref.coa_int_org_end_dt))))
     LEFT JOIN s_grnplm_vd_t_bvd_db_dmslcl.a_crncy_s s ON ((coa.crncy_id = s.crncy_id)))
     LEFT JOIN ( SELECT max(o.agrmnt_class_val_id) AS agrmnt_class_val_id
           FROM s_grnplm_as_t_didsd_010_vd_dwh.v_agrmnt_class_val o
          WHERE ((o.agrmnt_clsfctn_id = ANY (ARRAY['-1024'::integer, '-1025'::integer])) AND (upper(o.agrmnt_clsfctn_val_cd) = 'TYPE_КК'::text))) type1 ON ((1 = 1)));
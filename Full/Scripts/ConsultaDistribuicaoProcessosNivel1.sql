SELECT 'DIREITO DO TRABALHO' AS assunto_nivel_1, '864' AS cd_assunto_nivel_1, (SELECT count(*) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '864') AS qtd_sub_assuntos_total,  
(SELECT count(distinct cd_assunto_nivel_3) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '864' ) AS qtd_sub_assuntos_nivel_3
UNION ALL
SELECT 'DIREITO ADMINISTRATIVO E OUTRAS MATÉRIAS DE DIREITO PÚBLICO' AS assunto_nivel_1 , '9985' AS cd_assunto_nivel_1 , (SELECT count(*) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '9985') AS qtd_sub_assuntos_total,  
(SELECT count(distinct cd_assunto_nivel_3) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '9985' ) AS qtd_sub_assuntos_nivel_3
UNION ALL
SELECT 'DIREITO CIVIL' AS assunto_nivel_1 , '899' AS cd_assunto_nivel_1 , (SELECT count(*) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '899') AS qtd_sub_assuntos_total,  
(SELECT count(distinct cd_assunto_nivel_3) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '899' ) AS qtd_sub_assuntos_nivel_3
UNION ALL
SELECT 'DIREITO INTERNACIONAL'  AS assunto_nivel_1, '6191'  AS cd_assunto_nivel_1, (SELECT count(*) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '6191') AS qtd_sub_assuntos_total,  
(SELECT count(distinct cd_assunto_nivel_3) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '6191' ) AS qtd_sub_assuntos_nivel_3
UNION ALL
SELECT 'DIREITO PROCESSUAL CIVIL E DO TRABALHO'  AS assunto_nivel_1, '8826' AS cd_assunto_nivel_1 , (SELECT count(*) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '8826') AS qtd_sub_assuntos_total,  
(SELECT count(distinct cd_assunto_nivel_3) FROM pje_pte.vs_nivel_assunto WHERE cd_assunto_nivel_1 = '8826' ) AS qtd_sub_assuntos_nivel_3

--Logar na base padrao de segundo grau
--Criar ligacao com a base de log
--Colher percentual de retificacao total
--Verificar percentual de processos elegiveis (que contem apenas um RO) foram retificados
--Verificar quantos processos elegiveis (que contem apenas um RO) foram reificados

---------------------------------------------------------------------------------------------------
--TRT X - 2 GRAU
---------------------------------------------------------------------------------------------------

--Configurando acesso a base log
CREATE EXTENSION postgres_fdw;

CREATE SERVER foreign_server
        FOREIGN DATA WRAPPER postgres_fdw
        OPTIONS (host '10.0.3.150', port '5XX2', dbname 'pje_2grau_bugfix_log');
        
CREATE USER MAPPING FOR bugfix
        SERVER foreign_server
        OPTIONS (user '*****', password '*****');
       

CREATE FOREIGN TABLE tb_log_fdw (
        ds_entidade text,
        ds_id_entidade text,
        dt_log timestamp
)
        SERVER foreign_server
        OPTIONS (schema_name 'pje', table_name 'tb_log');
       
--Colhendo estatisticas       
--Quantitade de assuntos retificados (nao é a quantidade de processos....)
select count(*) from tb_log_fdw where ds_entidade ilike '%ProcessoAssunto%';

--Periodo de armazenamento de log
select  min(dt_log) as primeiro_registro, 
max(dt_log) as ultimo_registro, 
extract(year from age(max(dt_log), min(dt_log))) * 12 + extract(month from age(max(dt_log), min(dt_log))) as meses 
from tb_log_fdw
--3

--Quantidade de processos: 
select count(*) from tb_processo;
--287.041

--Quantidade de processos retificados
select 'Quantidade total de processos' , count(*) from pje.tb_processo_trf 
--where dt_autuacao < '2019-08-15'
--281965
union ALL
--Quantidade de processos retificados
select 'Quantidade de processos retificados' , count (distinct(id_processo_trf)) from pje.tb_processo_assunto where id_processo_assunto in 
	(select ds_id_entidade::int from tb_log_fdw where ds_entidade ilike '%ProcessoAssunto%')
--25854
union ALL
--Quantidade de processos elegíveis
SELECT 'Quantidade de processos elegíveis' ,count(id_processo) FROM  
		-- QUANTIDADE DE PROCESSOS QUE POSSUEM APENAS UM RECURSO ORDINARIO
		(SELECT id_processo, count(id_processo_documento) as total_recursos_ordinarios 
		 FROM tb_processo_documento
		 WHERE id_tipo_processo_documento = (select id_tipo_processo_documento from tb_tipo_processo_documento where ds_tipo_processo_documento = 'Recurso Ordinário')
		 --AND dt_juntada < '2019-08-15' 
		GROUP BY id_processo
		HAVING count(id_processo_documento) = 1) as t
--155918
union ALL
--Quantidade de processos elegiveis (que contem apenas um RO) foram reificados
select 'Quantidade de processos elegíveis retificados' ,count (distinct(id_processo_trf)) from pje.tb_processo_assunto 
where id_processo_assunto in (select ds_id_entidade::int from tb_log_fdw where ds_entidade ilike '%ProcessoAssunto%') 
and id_processo_trf IN 
		(SELECT id_processo FROM  
			-- QUANTIDADE DE PROCESSOS QUE POSSUEM APENAS UM RECURSO ORDINARIO
			(SELECT id_processo, count(id_processo_documento) as total_recursos_ordinarios 
			 FROM tb_processo_documento
			 WHERE id_tipo_processo_documento = (select id_tipo_processo_documento from tb_tipo_processo_documento where ds_tipo_processo_documento = 'Recurso Ordinário')
			 --AND dt_juntada < '2019-08-15' 
			GROUP BY id_processo
			HAVING count(id_processo_documento) = 1) AS t
		)
--13682
		
		

DROP FOREIGN TABLE IF EXISTS tb_log_fdw;
DROP USER MAPPING IF EXISTS FOR bugfix SERVER foreign_server;
DROP SERVER IF EXISTS foreign_server;
DROP EXTENSION IF EXISTS postgres_fdw;

CREATE EXTENSION postgres_fdw;

CREATE SERVER foreign_server_2
        FOREIGN DATA WRAPPER postgres_fdw
        OPTIONS (host '****', port '***', dbname '****');

CREATE USER MAPPING FOR bugfix
        SERVER foreign_server_2
        OPTIONS (user '***', password '***');

CREATE FOREIGN TABLE tb_processo_assunto_fdw (
        id_processo_assunto int,
        id_processo_trf int,
        id_assunto_trf int
)
        SERVER foreign_server_2
        OPTIONS (schema_name 'pje', table_name 'tb_processo_assunto');


CREATE FOREIGN TABLE tb_processo_fdw(
        id_processo int,
        nr_processo text
)
        SERVER foreign_server_2
        OPTIONS (schema_name 'pje', table_name 'tb_processo');

CREATE FOREIGN TABLE tb_assunto_trf_fdw(
        id_assunto_trf int,
        cd_assunto_trf text
)
        SERVER foreign_server_2
        OPTIONS (schema_name 'pje', table_name 'tb_assunto_trf');



with primeirograu as (
select p1g.nr_processo as processo_1g, a1g.cd_assunto_trf as cd_assunto_1g
from tb_processo_assunto pa1g
inner join pje.tb_assunto_trf a1g on a1g.id_assunto_trf = pa1g.id_assunto_trf
inner join tb_processo p1g on p1g.id_processo = pa1g.id_processo_trf
where p1g.id_processo in (select id_processo_trf from pje.tb_manifestacao_processual where cd_origem ilike '%envio')),
segundograu as (
select p2g.nr_processo as processo_2g, a2g.cd_assunto_trf as cd_assunto_2g
from tb_processo_assunto_fdw pa2g
inner join tb_assunto_trf_fdw a2g on a2g.id_assunto_trf = pa2g.id_assunto_trf
inner join tb_processo_fdw p2g on p2g.id_processo = pa2g.id_processo_trf)
select p1.processo_1g as processo, cd_assunto_1g as cd_assunto, '1grau' as grau
from primeirograu p1
union all
select p2.processo_2g, cd_assunto_2g, '2 grau' as grau
from segundograu p2 where exists (select p1.processo_1g from primeirograu p1 where p1.processo_1g = p2.processo_2g)

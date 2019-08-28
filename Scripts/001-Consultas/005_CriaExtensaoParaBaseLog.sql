DROP FOREIGN TABLE IF EXISTS tb_log_fdw;
DROP USER MAPPING IF EXISTS FOR bugfix SERVER foreign_server;
DROP SERVER IF EXISTS foreign_server;
/*DROP EXTENSION IF EXISTS postgres_fdw;
CREATE EXTENSION postgres_fdw;*/

CREATE SERVER foreign_server
        FOREIGN DATA WRAPPER postgres_fdw
        OPTIONS (host '{0}', port '{1}', dbname '{2}');

CREATE USER MAPPING FOR bugfix
        SERVER foreign_server
        OPTIONS (user '{3}', password '{4}');


CREATE FOREIGN TABLE tb_log_fdw (
        ds_entidade text,
        ds_id_entidade text,
        dt_log timestamp
)
        SERVER foreign_server
        OPTIONS (schema_name 'pje', table_name 'tb_log');


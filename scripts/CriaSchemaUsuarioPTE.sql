/* SEQUENCIA DE SCRIPTS QUE:
 * 
 * 1) Cria usuário, role e schema pje_pte
 * 2) Atualiza search_path
 * 3) Cria tabela tb_pte_processo_documento_indexacao (deprecated...)
 * 4) Cria a visão do pte
 * 5) Cria a view so de assuntos para facilitar o acesso à árvore
 */


--##################################################################################################
--##################################################################################################
-- PJE_2.3.0_066__DDL_PJEKZ-2373_CREATE_USER_PJE_USUARIO_SERVICO_PTE
--##################################################################################################
--##################################################################################################
/*
* Objetivo         : Cria usuario de servico para o módulo Pesquisa Textula (PTE).
* Issue            : [PJEKZ-2373]
* Sistema Satelite : GIGS
* Descricao        : Criar o usuario pje_usuario_servico_pte
*
* Equipe          : SMPAD
* Issue           : [PJEKZ-2373]
* Autor           : Ana Carolina Pereira Rocha Martins
* Email           : acprocha@tst.jus.br
* Data criação    : 23/05/2018
* 
*/
-- INICIO - Funcoes auxiliares --
CREATE OR REPLACE FUNCTION public.fn_pjekz_2373_usuario_nao_existe(nome_usuario varchar)
RETURNS boolean AS $BODY$
DECLARE
    var_exists integer;

BEGIN
    SELECT 1 INTO var_exists
    FROM pg_catalog.pg_roles
    WHERE rolname = nome_usuario;
    return (var_exists is null);
END;
$BODY$ LANGUAGE plpgsql;

-- Funcao principal
CREATE OR REPLACE FUNCTION public.fn_pjekz_2373_executa()
RETURNS void AS $BODY$

BEGIN
  IF public.fn_pjekz_2373_usuario_nao_existe('pje_usuario_servico_pte') THEN
    CREATE ROLE pje_usuario_servico_pte LOGIN
      ENCRYPTED PASSWORD 'md53cc5a42a68ccbd71b4455033272f96a2'
      NOSUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION;
    COMMENT ON role pje_usuario_servico_pte IS 'Usuário Servico : PJe Pesquisa Textual';

  END IF;

END; $BODY$
LANGUAGE plpgsql;

-- Execucao da funcao principal
SELECT public.fn_pjekz_2373_executa();

-- Exclusao das funcoes criadas
DROP FUNCTION public.fn_pjekz_2373_executa();
DROP FUNCTION public.fn_pjekz_2373_usuario_nao_existe(nome_usuario varchar);


--##################################################################################################
--##################################################################################################
-- PJE_2.3.0_067__DDL_PJEKZ-2373_CREATE_USER_PJE_PTE
--##################################################################################################
--##################################################################################################

/*
* Objetivo         : Cria usuario de servico para o módulo Pesquisa Textula (PTE).
* Issue            : [PJEKZ-2373]
* Sistema Satelite : GIGS
* Descricao        : Criar o usuario pje_pte
*
* Equipe          : SMPAD
* Issue           : [PJEKZ-2373]
* Autor           : Ana Carolina Pereira Rocha Martins
* Email           : acprocha@tst.jus.br
* Data criação    : 23/05/2018
* 
*/
-- INICIO - Funcoes auxiliares --
CREATE OR REPLACE FUNCTION public.fn_pjekz_2373_usuario_nao_existe(nome_usuario varchar)
RETURNS boolean AS $BODY$
DECLARE
    var_exists integer;

BEGIN
    SELECT 1 INTO var_exists
    FROM pg_catalog.pg_roles
    WHERE rolname = nome_usuario;
    return (var_exists is null);
END;
$BODY$ LANGUAGE plpgsql;

-- Funcao principal
CREATE OR REPLACE FUNCTION public.fn_pjekz_2373_executa()
RETURNS void AS $BODY$

BEGIN
  IF public.fn_pjekz_2373_usuario_nao_existe('pje_pte') THEN
    CREATE ROLE pje_pte LOGIN
      ENCRYPTED PASSWORD 'md53cc5a42a68ccbd71b4455033272f96a2'
      NOSUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION;
    COMMENT ON role pje_pte IS 'Usuário PJe Pesquisa Textual';

  END IF;

END; $BODY$
LANGUAGE plpgsql;

-- Execucao da funcao principal
SELECT public.fn_pjekz_2373_executa();

-- Exclusao das funcoes criadas
DROP FUNCTION public.fn_pjekz_2373_executa();
DROP FUNCTION public.fn_pjekz_2373_usuario_nao_existe(nome_usuario varchar);


--##################################################################################################
--##################################################################################################
-- PJE_2.3.0_068__DDL_PJEKZ-2373_CREATE_ROLE_PTERO_USUARIO_SERVICO
--##################################################################################################
--##################################################################################################
/*
* Objetivo         : Cria role para o usuario de módulo Pesquisa Textual (PTE)
* Issue            : [PJEKZ-2373]
* Módulo           : PTE
* Descricao        : Criar a role pjero_usuario_servico_pte
*
* Equipe          : SMPAD
* Issue           : [PJEKZ-2373]
* Autor           : Ana Carolina Pereira Rocha Martins
* Email           : acprocha@tst.jus.br
* Data criação    : 23/05/2018
* 
*/

-- INICIO - Funcoes auxiliares --
CREATE OR REPLACE FUNCTION public.fn_pjekz_2373_executa()
RETURNS void AS $BODY$
DECLARE
    v_verifica_existencia smallint;

BEGIN
    v_verifica_existencia := 0;
    SELECT count(*) INTO v_verifica_existencia
    FROM pg_roles
    WHERE UPPER(rolname)='PJERO_USUARIO_SERVICO_PTE'
    AND rolcanlogin is false;

    IF v_verifica_existencia = 0 THEN
      CREATE ROLE pjero_usuario_servico_pte;
      COMMENT ON role pjero_usuario_servico_pte IS 'Módulo: PJe Pesquisa Textual';

      -- Associacao da role com o usuario de servico do Pesquisa Textual
      GRANT pjero_usuario_servico_pte TO pje_usuario_servico_pte;

    END IF;

END;
$BODY$ LANGUAGE plpgsql;

-- Execucao da funcao principal
SELECT public.fn_pjekz_2373_executa();

-- Exclusao das funcoes criadas
DROP FUNCTION public.fn_pjekz_2373_executa();

--##################################################################################################
--##################################################################################################
-- PJE_2.3.0_069__DDL_PJEKZ-2373_CRIA_SCHEMA_PJE_PTE
--##################################################################################################
--##################################################################################################
/*
* Objetivo         : Criar schema próprio para o módulo Pesquisa Textual.
* Issue            : [PJEKZ-2373]
* Sistema Satelite : Pesquisa Textual
* Descricao        : Criar schema próprio para o módulo Pesquisa Textual.
*
* Equipe          : SMPAD
* Issue           : [PJEKZ-2373]
* Autor           : Ana Carolina Pereira Rocha Martins
* Email           : acprocha@tst.jus.br
* Data criação    : 23/05/2018
* 
*/

-- INICIO - Funcoes auxiliares --
CREATE OR REPLACE FUNCTION public.fn_pjekz_2373_executa()
RETURNS void AS $BODY$
DECLARE
	v_verifica_existencia int := 0;
BEGIN

    
    select count(*) into v_verifica_existencia from pg_catalog.pg_namespace where UPPER(nspname) = 'PJE_PTE';
    IF v_verifica_existencia = 0 THEN
        CREATE SCHEMA pje_pte AUTHORIZATION pje;
        GRANT USAGE ON SCHEMA pje_pte TO pjero_usuario_servico_pte;
		
		COMMENT ON SCHEMA pje_pte IS 'PJE-JT';		
	END IF;	

END;
$BODY$ LANGUAGE plpgsql;

-- Execucao da funcao principal
SELECT public.fn_pjekz_2373_executa();

-- Exclusao das funcoes criadas
DROP FUNCTION public.fn_pjekz_2373_executa();

--##################################################################################################
--##################################################################################################
-- PJE_2.3.0_070__DDL_PJEKZ-2373_GRANTS_PJE_ROLE_PTERO_USUARIO_SERVICO
--##################################################################################################
--##################################################################################################
/*
* Objetivo         : Atribuir grants para os schemas.
* Issue            : [PJEKZ-2373]
* Sistema Satelite : Pesquisa Textual
* Descricao        : Atribuir grants para os schemas e os objetos
*
* Equipe          : SMPAD
* Issue           : [PJEKZ-2373]
* Autor           : Ana Carolina Pereira Rocha Martins
* Email           : acprocha@tst.jus.br
* Data criação    : 23/05/2018
* 
*/

-- INICIO - Funcoes auxiliares --
CREATE OR REPLACE FUNCTION public.fn_pjekz_2373_executa()
RETURNS void AS $BODY$

BEGIN

    -- Autorizacao da role do GIGS para fazer acesso aos schemas
    GRANT USAGE ON SCHEMA pje                           TO pjero_usuario_servico_pte;
    GRANT USAGE ON SCHEMA pje_pte                       TO pjero_usuario_servico_pte;
    GRANT USAGE ON SCHEMA pje_pte                       TO pjero_usuario_servico;

    -- Autorizacao de acesso de leitura nas tabelas para a role do PTE
    GRANT SELECT ON pje.tb_processo_documento           TO pjero_usuario_servico_pte;
    GRANT SELECT ON pje.tb_processo_documento_bin       TO pjero_usuario_servico_pte;
    GRANT SELECT ON pje.tb_proc_doc_bin_pess_assin      TO pjero_usuario_servico_pte;
    GRANT SELECT ON pje.tb_usuario_login                TO pjero_usuario_servico_pte;
    GRANT SELECT ON pje.tb_processo                     TO pjero_usuario_servico_pte;
    GRANT SELECT ON pje.tb_processo_trf                 TO pjero_usuario_servico_pte;
    GRANT SELECT ON pje.tb_tipo_processo_documento      TO pjero_usuario_servico_pte;
    GRANT SELECT ON pje.tb_orgao_julgador               TO pjero_usuario_servico_pte;
    GRANT SELECT ON pje.tb_orgao_julgador_colgiado      TO pjero_usuario_servico_pte;
    
END;
$BODY$ LANGUAGE plpgsql;

-- Execucao da funcao principal
SELECT public.fn_pjekz_2373_executa();

-- Exclusao das funcoes criadas
DROP FUNCTION public.fn_pjekz_2373_executa();


--##################################################################################################
--##################################################################################################
-- SCRIPT PJE_2.3.0_905__DDL_PJEKZ-2373_ALTERACAO_DO_SEARCH_PATH
--##################################################################################################
--##################################################################################################

/*
 * Objetivo      : Alteração da variável "search_path" no database para acrescentar o schema pje_pte
 * Issue         : [PJEKZ-2373]
 * SubSistema    : (X)1ºGrau (X)2ºGrau (X)3ºGrau
 * Descricao     : Alteração da variável "search_path" no database para acrescentar o schema pje_pte
 *
 * Equipe        : Administração de Banco de Dados (SMPAD-BD)
 * Issue         : [PJEKZ-2373]
 * Avaliador     : Ana Carolina Pereira Rocha Martins
 * Email         : acprocha@tst.jus.br
 * Data avaliacao: 22/05/2018
 *
*/

DO
$BODY$
DECLARE
  vcurrent_database varchar;

BEGIN

  vcurrent_database := current_database();
  
  EXECUTE 'ALTER DATABASE ' || vcurrent_database || ' SET search_path = "$user", public, pje, pje_adm, pje_stage, pje_jbpm, pje_jt, pje_mnt, pje_qrtz, pje_util, pje_gim, pje_pte';
  
END;
$BODY$ LANGUAGE plpgsql;


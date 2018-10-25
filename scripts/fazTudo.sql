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
-- SCRIPT PJE_2.3.0_900__DDL_PJEKZ-2373_CREATE_USER_PJE_USUARIO_SERVICO_PTE
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
-- SCRIPT PJE_2.3.0_901__DDL_PJEKZ-2373_CREATE_USER_PJE_PTE
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
-- SCRIPT PJE_2.3.0_902__DDL_PJEKZ-2373_CREATE_ROLE_PTERO_USUARIO_SERVICO
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
      GRANT pjero_usuario_servico_pte TO pje_pte;

    END IF;

END;
$BODY$ LANGUAGE plpgsql;

-- Execucao da funcao principal
SELECT public.fn_pjekz_2373_executa();

-- Exclusao das funcoes criadas
DROP FUNCTION public.fn_pjekz_2373_executa();

--##################################################################################################
--##################################################################################################
-- SCRIPT PJE_2.3.0_903__DDL_PJEKZ-2373_CRIA_SCHEMA_PJE_PTE
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
        CREATE SCHEMA pje_pte AUTHORIZATION pje_pte;
        GRANT USAGE ON SCHEMA pje_pte TO pjero_usuario_servico_pte;
		
		COMMENT ON SCHEMA pje_pte IS 'PJE';		
	END IF;	

	

END;
$BODY$ LANGUAGE plpgsql;

-- Execucao da funcao principal
SELECT public.fn_pjekz_2373_executa();

-- Exclusao das funcoes criadas
DROP FUNCTION public.fn_pjekz_2373_executa();

--##################################################################################################
--##################################################################################################
-- SCRIPT PJE_2.3.0_904__DDL_PJEKZ-2373_GRANTS_PJE_ROLE_PTERO_USUARIO_SERVICO
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

    -- Autorizacao de acesso de leitura nas tabelas para a role do GIGS
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

--##################################################################################################
--##################################################################################################
-- SCRIPT PJE_2.3.0_907__DDL_PJEKZ-3772_CRIAR_TB_PTE_PROCESSO_DOCUMENTO_INDEXACAO
--##################################################################################################
--##################################################################################################

/*
 * Objetivo      : Criação da tabela tb_pte_processo_documento_indexacao
 * Issue         : [PJEKZ-XXXX]
 * SubSistema    : (X)1ºGrau (X)2ºGrau (X)3ºGrau
 * Descricao     : Criação da tabela tb_pte_processo_documento_indexacao. Esta opção foi escolhida para melhorar a performance do extrator, 
 * 					uma vez que o postgres não precisará fazer a criação e inativacao de registros de toda a tabela tb_processo_documento. 
 *
 * Equipe        : CTPJE/SMPAD
 * Issue         : [PJEKZ-XXXX]
 * Autor         : Ana Carolina Pereira Rocha Martins
 * Email		 : acprocha@tst.jus.br
 * Data Criacao  : 20/06/2017 * 
 * 
*/


CREATE OR REPLACE FUNCTION public.fn_pjekz_3772_tabela_nao_existe(nome_schema varchar, nome_tabela varchar) RETURNS boolean AS $$
  DECLARE
    var_table_exists smallint;

BEGIN
        SELECT 1 INTO var_table_exists
        FROM information_schema.TABLES
        WHERE table_schema = nome_schema
        AND table_name     = nome_tabela;
    
        return (var_table_exists is null);
END; $$
 LANGUAGE plpgsql;

-- Funcao principal -- Inicio
CREATE OR REPLACE FUNCTION public.fn_pjekz_3772_executa() RETURNS void AS $$
BEGIN

	IF  (public.fn_pjekz_3772_tabela_nao_existe('pje', 'tb_pte_processo_documento_indexacao')) THEN
	
		/*==============================================================*/
		/* Table:  tb_pte_processo_documento_indexacao                  */
		/*==============================================================*/
		create table pje.tb_pte_processo_documento_indexacao (
		   id_processo_documento 		  integer              NOT NULL,
		   in_documento_indexado          CHAR(1)              NOT NULL DEFAULT 'S',
		   constraint tb_pte_processo_documento_indexacao_cc01 check (in_documento_indexado in ('S')),
		   constraint tb_pte_processo_documento_indexacao_pk primary key (id_processo_documento)
		);

		comment on table pje.tb_pte_processo_documento_indexacao is
		'Tabela que armazena a informação se um documento está indexado';

		comment on column pje.tb_pte_processo_documento_indexacao.id_processo_documento is
		'Número do id_processo_documento proveniente da tb_processo_documento.';

		comment on column pje.tb_pte_processo_documento_indexacao.in_documento_indexado is
		'Informa se o documento está indexado (S). Este campo foi criado desta forma pois, embora hoje so se tenha uma opção - S -, é possível que no futuro mais opções venham a existir.';

		-- set table ownership
		alter table pje.tb_pte_processo_documento_indexacao owner to pje
		;
		/*==============================================================*/
		/* Index: tb_pte_processo_documento_indexacao_in01                             */
		/*==============================================================*/
		create  index tb_pte_processo_documento_indexacao_in01 on pje.tb_pte_processo_documento_indexacao using BTREE (
		in_documento_indexado
		);


		grant DELETE,INSERT,SELECT,UPDATE on pje.tb_pte_processo_documento_indexacao to pjero_usuario_manutencao;

		grant DELETE,INSERT,SELECT,UPDATE on pje.tb_pte_processo_documento_indexacao to pjero_usuario_servico;

		alter table pje.tb_pte_processo_documento_indexacao
		   add constraint tb_pte_processo_documento_indexacao_fk01 foreign key (id_processo_documento)
			  references pje.tb_processo_documento (id_processo_documento)
			  on delete restrict on update restrict;
			  
		GRANT ALL ON table pje.tb_pte_processo_documento_indexacao TO pje;
	END IF;
	
END; $$
LANGUAGE plpgsql;
-- Funcao principal -- Fim
SELECT public.fn_pjekz_3772_executa();

DROP FUNCTION public.fn_pjekz_3772_executa();
DROP FUNCTION public.fn_pjekz_3772_tabela_nao_existe(nome_schema varchar, nome_tabela varchar);

--##################################################################################################
--##################################################################################################
-- SCRIPT PJE_2.3.0_906__DDL_PJEKZ-2728_CRIAR_VIEW_VS_DOCUMENTO_INDEXAVEL
--##################################################################################################
--##################################################################################################

/*
 * Objetivo      : Criação da view vs_documento_indexavel
 * Issue         : [PJEKZ-2728]
 * SubSistema    : (X)1ºGrau (X)2ºGrau (X)3ºGrau
 * Descricao     : Criação da view vs_documento_indexavel para listar informações dos documentos não indexados
 *
 * Equipe        : CTPJE/SMPAD
 * Issue         : [PJEKZ-2728]
 * Autor         : Ana Carolina Pereira Rocha Martins
 * Email		 : acprocha@tst.jus.br
 * Data Criacao  : 21/05/2017 * 
 * 
*/

CREATE OR REPLACE FUNCTION public.fn_pjekz_2728_executa()
RETURNS void AS $BODY$
BEGIN

	DROP VIEW IF EXISTS pje_pte.vs_documento_indexavel;

	/*==============================================================*/
	/* View: vs_documento_indexavel              */
	/*==============================================================*/

	CREATE OR REPLACE VIEW pje_pte.vs_documento_indexavel AS
		 WITH grau AS (
		         SELECT tb_parametro.vl_variavel
		           FROM tb_parametro
		          WHERE tb_parametro.nm_variavel::text = 'aplicacaoSistema'::text
		        ), tribunal AS (
		         SELECT tb_tribunal.cd_sigla_tribunal
		           FROM tb_tribunal
		        )
		 SELECT doc.id_processo_documento,
		    doc.id_processo_documento_bin,
		    doc.ds_identificador_unico,
		    docbin.ds_extensao,
		    doc.ds_processo_documento,
		    doc.id_tipo_processo_documento,
		    tpd.cd_documento,
		    tpd.ds_tipo_processo_documento,
		    assin.id_pessoa,
		    ul.ds_nome,
		    doc.dt_juntada,
		    docbin.ds_modelo_documento as tx_conteudo_documento,
		    --.in_documento_indexado,
		    ( SELECT tribunal.cd_sigla_tribunal
		           FROM tribunal) AS cd_tribunal,
		    doc.ds_instancia AS nr_grau,
		    (SELECT ds_identificador_unico FROM tb_processo_documento doc2 WHERE doc2.id_processo_documento = doc.id_documento_principal) AS ds_identificador_unico_documento_principal,
		    p.nr_processo,
		    doc.id_processo,
		    ptrf.id_orgao_julgador,
		    oj.ds_orgao_julgador,
		    ptrf.id_orgao_julgador_colegiado,
		    ojc.ds_orgao_julgador_colegiado,
		    doc.in_documento_sigiloso,
		    ptrf.in_segredo_justica,
		    a.cd_assunto_trf,
		    a.ds_assunto_trf,
		    a.ds_assunto_completo
		   FROM tb_processo_documento doc
		     JOIN tb_processo_documento_bin docbin ON docbin.id_processo_documento_bin::integer = doc.id_processo_documento_bin::integer 
		     JOIN tb_proc_doc_bin_pess_assin assin ON assin.id_processo_documento_bin::integer = docbin.id_processo_documento_bin::integer
		     JOIN tb_usuario_login u ON u.id_usuario = assin.id_pessoa::integer
		     JOIN tb_processo p ON p.id_processo::integer = doc.id_processo::integer 
		     JOIN tb_processo_trf ptrf ON ptrf.id_processo_trf::integer = p.id_processo::integer
		     JOIN tb_tipo_processo_documento tpd ON tpd.id_tipo_processo_documento::integer = doc.id_tipo_processo_documento::integer
		     JOIN tb_usuario_login ul ON ul.id_usuario = assin.id_pessoa::integer
		     LEFT JOIN tb_orgao_julgador oj ON ptrf.id_orgao_julgador::integer = oj.id_orgao_julgador::integer
		     LEFT JOIN tb_orgao_julgador_colgiado ojc ON  ojc.id_orgao_julgador_colegiado::integer = ptrf.id_orgao_julgador_colegiado::integer
		     LEFT JOIN tb_processo_assunto pa ON pa.id_processo_trf::integer = ptrf.id_processo_trf::integer
		     LEFT JOIN tb_assunto_trf a ON a.id_assunto_trf = pa.id_assunto_trf::integer
		  WHERE doc.dt_juntada IS NOT NULL 
		  	AND doc.ds_instancia::text = (( SELECT grau.vl_variavel FROM grau)) 
		  	AND ptrf.cd_processo_status = 'D'::bpchar
		    AND pa.in_assunto_principal = 'S'
		    AND doc.in_documento_sigiloso = 'N'
		    AND ptrf.in_segredo_justica = 'N';
           

	ALTER TABLE pje_pte.vs_documento_indexavel
	  OWNER TO pje_pte;
	  
	GRANT SELECT ON TABLE pje_pte.vs_documento_indexavel TO pjero_usuario_servico_pte;

	COMMENT ON VIEW pje_pte.vs_documento_indexavel
	  IS 'View utilizada pelo Módulo Pesquisa Textual para listar informações dos documentos não indexados';
	  
	      -- Autorizacao da role do GIGS para fazer acesso aos schemas
    GRANT USAGE ON SCHEMA pje                           TO pjero_usuario_servico_pte;

    -- Autorizacao de acesso de leitura nas tabelas para a role do GIGS
    GRANT SELECT ON pje.tb_processo_documento           TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_processo_documento_bin       TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_proc_doc_bin_pess_assin      TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_usuario_login                TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_processo                     TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_processo_trf                 TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_tipo_processo_documento      TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_orgao_julgador               TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_orgao_julgador_colgiado      TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_processo_assunto				TO pje_pte WITH GRANT OPTION;
    GRANT SELECT ON pje.tb_assunto_trf					TO pje_pte WITH GRANT OPTION;
	GRANT SELECT ON pje.tb_parametro					TO pje_pte WITH GRANT OPTION;
	GRANT SELECT ON pje.tb_tribunal						TO pje_pte WITH GRANT OPTION;
	
END;
$BODY$ LANGUAGE plpgsql;

SELECT public.fn_pjekz_2728_executa();

DROP FUNCTION public.fn_pjekz_2728_executa();

--##################################################################################################
--##################################################################################################
-- SCRIPT PARA CRIAR A VIEW
--##################################################################################################
--##################################################################################################
CREATE OR REPLACE VIEW vs_nivel_assunto AS 
SELECT b.*,
CASE 
		WHEN b.nivel = '2' THEN 
			(SELECT pai.cd_assunto_trf FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))
		WHEN b.nivel = '3' THEN 
			(SELECT avo.cd_assunto_trf FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf)))
		WHEN b.nivel = '4' THEN
			(SELECT bisavo.cd_assunto_trf FROM tb_assunto_trf bisavo WHERE bisavo.id_assunto_trf =
			(SELECT avo.id_assunto_trf_superior FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))))
		WHEN b.nivel = '5' THEN
			(SELECT tataravo.cd_assunto_trf FROM tb_assunto_trf tataravo WHERE tataravo.id_assunto_trf =
			(SELECT bisavo.id_assunto_trf_superior FROM tb_assunto_trf bisavo WHERE bisavo.id_assunto_trf =
			(SELECT avo.id_assunto_trf_superior FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf)))))
	ELSE b.cd_assunto_trf END  as cd_assunto_nivel_1,
CASE 
		WHEN b.nivel = '1' THEN  ''
		WHEN b.nivel = '3' THEN 
			(SELECT pai.cd_assunto_trf FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))
		WHEN b.nivel = '4' THEN 
			(SELECT avo.cd_assunto_trf FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf)))
		WHEN b.nivel = '5' THEN
			(SELECT bisavo.cd_assunto_trf FROM tb_assunto_trf bisavo WHERE bisavo.id_assunto_trf =
			(SELECT avo.id_assunto_trf_superior FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))))
	ELSE b.cd_assunto_trf END  as cd_assunto_nivel_2,
CASE 
		WHEN b.nivel = '1' THEN  ''
		WHEN b.nivel = '2' THEN  ''
		WHEN b.nivel = '4' THEN 
			(SELECT pai.cd_assunto_trf FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))
		WHEN b.nivel = '5' THEN 
			(SELECT avo.cd_assunto_trf FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf)))
	ELSE b.cd_assunto_trf END  as cd_assunto_nivel_3,
CASE 
		WHEN b.nivel = '1' THEN  ''
		WHEN b.nivel = '2' THEN  ''
		WHEN b.nivel = '3' THEN  ''
		WHEN b.nivel = '5' THEN 
			(SELECT pai.cd_assunto_trf FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))
	ELSE b.cd_assunto_trf END  as cd_assunto_nivel_4,
CASE 
		WHEN b.nivel = '1' THEN  ''
		WHEN b.nivel = '2' THEN  ''
		WHEN b.nivel = '3' THEN  ''
		WHEN b.nivel = '4' THEN  ''		
	ELSE b.cd_assunto_trf END  as cd_assunto_nivel_5
FROM (
SELECT a.*, 
		CASE WHEN a.cd_assunto_trf in ('9985','899','864','6191','8826') THEN '1'
		WHEN a.cd_assunto_trf in ('10186','10186','9997','10370','10421','10645','10394','10088','10409','9986','10385','10954','10219','','7947','7681','10431','55054','2622','55006','7644','1654','55336','55341','55077','1695','1658','2662',
		'7628','10568','2581','2620','2567','55218','1937','1397','55078','6197','9565','6213','6218','6202','6215','6212','8893','8938','8828','9148','9192','8875','11781','8883','8842','55230','8960','9045','55301',
		'55581','55596','55619') THEN '2'
		WHEN a.cd_assunto_trf in ('10187','10894','10015','10011','10022','10009','9998','10020','10382','10379','10381','10384','10377','10380','10376','10378','10374','10372','10373','10383','10375','10371','10423','10430','10429','10422','10428','10426','10425','10424','10427','10889','10646','10402','10395',
		'10089','10097','10411','10410','11848','9988','11847','11845','11846','11849','11842','11843','10587','10393','10392','10388','10387','10386','10391','10389','10390','10958','10955','10956','10957','10254','10695','10287','10258','10250','10279','10313','10220','10288','10276',
		'4701','5632','7690','7694','9580','7691','7688','55055','55056','2624','55010','55021','5276','5277','7645','5278','55024','5279','5280','55031','55032','5281','55035','55036','55037','5288','5282','55039','55322','55040','55041','5284','55053','55043','55044','5286','5287','5289','5290',
		'5291','55045','5293','5292','55034','5294','55047','55048','5295','5296','55049','7646','55050','5297','55052','5299','5301','5272','2670','1806','2233','7647','55065','1816','1844','2409','55070','2029','2133','1957','2554','2421','5273','2537','55337','55338','55339','55340','55342',
		'55343','55344','55345','55347','55088','1690','1691','1773','10564','55348','1703','55091','55009','55087','4438','4435','55354','55008','1705','4452','4437','55355','1663','55108','55095','55105','2086','1661','2139','2140','2426','2116','55104','10581','2663','55113','55115','2019',
		'2021','2557','2559','55116','2558','55117','55118','7629','7631','55119','55120','55121','7630','55122','7633','7632','55123','55382','10571','55383','55384','55385','55386','55387','10570','55225','10569','2583','2594','2606','55148','55149','55397','1767','1783','55150','1789','2349',
		'2666','1888','1920','2450','2055','55159','8813','2215','55170','2273','2364','2331','55172','4442','55169','2458','2477','55400','2117','2493','2506','2540','55192','1849','1904','55196','55195','55197','55198','55403','8824','55199','1907','55202','2243','55204','55405','2656','2435',
		'2478','2546','55209','8808','1855','55216','55219','55220','55423','5356','8807','8806','8805','2704','55080','55079','55081','55082','6201','10938','10939','8919','8928','8934','55254','55434','8942','55438','8941','55439','8939','8829','55456','8838','9520','9519','9517','9418','9163',
		'55277','9518','10683','55639','55459','9419','10880','9180','9414','9450','55462','9453','9160','10686','9166','9178','55281','10670','10671','10672','55470','10573','55471','9189','10673','9484','9149','9532','55288','9524','9196','55474','55475','55479','11786','11783','11785','11782',
		'11784','8884','8888','8843','9493','9258','8859','8866','8868','8873','9494','8867','8874','55232','55234','55235','8865','55237','55498','55233','55499','55236','8961','8986','55503','8990','9024','55296','9026','9098','9047','55518','9050','55537','55538','10666','55299','55298','10738',
		'9060','55297','55546','55547','55554','55556','55559','55560','55308','55304','55310','55307','55566','55567','55309','55303','55571','55305','55574','55302','55311','55578','55579','55306','55582','55583','55588','55589','55590','55591','55592','55593','55594','55595','55597',
		'55604','55610','55620','55622','55627','55634') THEN '3'
		WHEN a.cd_assunto_trf in ('10189','10191','10190','10192','10675','10193','10188','55000','55001','10928','55313','10019','10012','10013','10014','10025','10024','10023','10006','10396','10399','10400','10096','11870','10098','10099','10100','10101','10256','10878','10255','10257','10698','10273','10272',
		'10270','10267','10268','10259','10271','10262','10274','10264','10265','10260','10275','10261','10269','10266','10263','10252','10251','10253','10281','10280','10283','10282','10946','10225','10231','10227','10234','10223','10222','10238','10241','10235','10881','10239','10693',
		'10240','10236','10237','10224','10233','10230','10226','10229','10232','10228','10662','10294','10638','10730','10293','10303','10291','10292','10309','10308','10302','10244','10245','10304','10248','10246','10247','10306','10300','10296','10298','10301','10290','10295','10310',
		'10289','10305','10718','10323','10311','10221','10312','10299','11858','10307','10249','10497','10705','10297','10277','10278','4703','4706','7709','7710','7707','10592','7706','7708','7703','7705','7704','7711','7714','9602','9608','9593','9609','7700','7698','10582','4718','55314',
		'55012','55020','55013','55019','55011','55014','55315','55015','55017','55018','55022','55023','55312','55025','55316','55317','55026','55029','55030','55320','55321','55038','55323','55324','55042','55325','55326','55327','55328','55051','55329','55330','55057','2266','1814',
		'1822','55058','55059','55060','1932','2445','2523','55063','55064','1807','55066','55067','55068','55069','5352','5354','2031','2033','2037','55071','55072','55073','55074','9487','55346','55349','55352','55353','55092','55093','55094','55007','55089','55090','55356','55357','55358',
		'55109','55111','55359','55360','55361','55362','55363','55364','55106','55365','55366','55098','55102','55367','55368','55099','55100','55101','55097','55369','55370','55112','55371','55372','55373','55374','55376','55377','55378','55379','55380','55114','55381','55124','55126',
		'55128','55127','55388','55129','55144','55145','55389','55146','1666','55143','1681','55147','55142','2604','55396','55152','55154','55153','55155','55398','55156','55157','55151','55158','55375','55162','55163','55164','55165','55166','55167','8818','8817','55161','55160','55168',
		'55171','55173','55174','55175','55399','55179','55188','55189','55181','55180','55182','5269','55176','2275','55183','55184','2449','55185','55177','2461','1721','8812','2463','8810','2697','2466','2468','8816','55178','2452','55186','55401','55402','55193','55194','55200','55404',
		'55203','55406','55407','55408','2657','1976','1929','1965','2661','1977','55206','55205','1978','1981','1966','55207','2479','2480','2641','8820','8821','8822','1998','2210','2212','8823','8809','55210','55212','55211','2569','55215','1723','1724','55214','55415','55416','9051','55213',
		'55417','55418','55419','55420','55421','55422','55424','55425','55217','55427','55428','55241','55242','55243','10734','55240','55432','55433','55255','55256','10737','55246','10735','55436','55437','55641','55249','55253','55440','55245','10652','10654','10653','55258','10901',
		'55259','55260','55257','55457','55458','55265','55270','55271','55268','55269','55276','55267','55266','55640','55460','55279','55280','55463','55464','10677','10676','55465','10680','10679','55467','10885','10869','55468','10678','10861','10860','55472','10685','55286','55287',
		'10684','10687','55476','55480','55481','10660','10659','55289','55290','55482','55226','55483','55484','55485','55486','55487','55488','55489','55490','10658','10655','10656','10657','55491','55492','55500','55501','55502','55300','10668','55504','10940','55506','55507','55508',
		'55513','55514','55515','55516','55517','55519','55523','55524','55525','55529','55530','55539','55540','55541','55542','55543','55544','55545','10669','55548','55549','55552','55553','55555','55557','55561','55562','55563','55564','55565','55568','55569','55570','55572','55573',
		'55637','55638','55575','55576','55580','55584','55585','55586','55587','55598','55599','55600','55601','55602','55603','55605','55606','55607','55608','55609','55611','55612','55613','55614','55615','55616','55617','55618','55621','55623','55624','55625','55626','55628','55632',
		'55635','55636')
		 THEN '4'
		WHEN a.cd_assunto_trf in ('10700','10701','10875','10893','10883','10884','10699','55016','55318','55319','55027','55028','55061','55062','55075','55331','55332','55333','55334','55335','55076','55350','55351','55110','55107','55103',
		'55130','55131','55135','55390','55391','55392','55134','55393','55136','55140','55138','55394','55395','55139','55191','55409','55410','55411','55412','55208','55413','55414','55426','55429','55430','55431',
		'55247','55435','55441','55442','55443','55444','55445','55446','55447','55448','55449','55450','55451','55452','55453','55454','55455','55272','55274','55275','55273','55461','55466','55469','55473','55477',
		'55478','55228','55493','55494','55495','55496','55497','55505','55509','55510','55511','55512','55520','55521','55522','55526','55527','55528','55531','55532','55533','55534','55535','55536','55550','55551',
		'55558','55577','55629','55630','55631','55633') THEN '5'
		ELSE 'Nível não identificado' END AS nivel		
	FROM tb_assunto_trf a
	WHERE in_ativo = 'S'
) AS b;

ALTER TABLE vs_nivel_assunto
	  OWNER TO pje_pte;
	  

GRANT SELECT ON TABLE vs_nivel_assunto TO pjero_usuario_servico_pte;

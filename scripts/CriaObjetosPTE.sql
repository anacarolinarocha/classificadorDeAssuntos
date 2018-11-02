
--##################################################################################################
--##################################################################################################
-- PJE_2.3.0_072__DDL_PJEKZ-3772_CRIAR_TB_PTE_STATUS_DOCUMENTO_INDEXADO
--##################################################################################################
--##################################################################################################
/*
 * Objetivo      : Criação da tabela tb_pte_status_documento_indexado
 * Issue         : [PJEKZ-3772][PJEKZ-5130]
 * SubSistema    : (X)1ºGrau (X)2ºGrau (X)3ºGrau
 * Descricao     : Criação da tabela tb_pte_status_documento_indexado. Esta opção foi escolhida para melhorar a performance do extrator, 
 * 					uma vez que o postgres não precisará fazer a criação e inativacao de registros de toda a tabela tb_processo_documento.
 *					[PJEKZ-5130] Nesta issue, adicionou-se o campo de data e hora de processamento, que irá facilitar a extração incremental de documentos. 					 
 *
 * Equipe        : CTPJE/SMPAD
 * Issue         : [PJEKZ-3772][PJEKZ-5130]
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

	IF  (public.fn_pjekz_3772_tabela_nao_existe('pje_pte', 'tb_pte_status_documento_indexado')) THEN
	
		/*==============================================================*/
		/* Table:  tb_pte_status_documento_indexado                  */
		/*==============================================================*/
		create table pje_pte.tb_pte_status_documento_indexado (
		   id_processo_documento 		  integer              NOT NULL,
		   cd_status_documento          CHAR(1)              NOT NULL DEFAULT 'I',
		   dh_processamento     		TIMESTAMP            not null,
		   constraint tb_pte_status_documento_indexado_cc01 check (cd_status_documento in ('I')),
		   constraint tb_pte_status_documento_indexado_pk primary key (id_processo_documento)
		);

		comment on table pje_pte.tb_pte_status_documento_indexado is
		'Tabela que armazena a informações do status de um documento indexado';

		comment on column pje_pte.tb_pte_status_documento_indexado.id_processo_documento is
		'Número do id_processo_documento proveniente da tb_processo_documento.';

		comment on column pje_pte.tb_pte_status_documento_indexado.cd_status_documento is
		'Informa o status do documento. Até o momento, o único status possível é I-Indexado';
		
		comment on column pje_pte.tb_pte_status_documento_indexado.dh_processamento is
		'Data e hora em que o documento foi processado';
		
		/*==============================================================*/
		/* Index: tb_pte_log_extracao_in01                              */
		/*==============================================================*/
		create  index tb_pte_status_documento_indexado_in01 on pje_pte.tb_pte_status_documento_indexado (
		dh_processamento
		);
		
		-- set table ownership
		alter table pje_pte.tb_pte_status_documento_indexado owner to pje;
	
		grant DELETE,INSERT,SELECT,UPDATE on pje_pte.tb_pte_status_documento_indexado to pjero_usuario_manutencao;

		grant DELETE,INSERT,SELECT,UPDATE on pje_pte.tb_pte_status_documento_indexado to pjero_usuario_servico;

		alter table pje_pte.tb_pte_status_documento_indexado
		   add constraint tb_pte_status_documento_indexado_fk01 foreign key (id_processo_documento)
			  references pje.tb_processo_documento (id_processo_documento)
			  on delete restrict on update restrict;
		  
	END IF;
	
END; $$
LANGUAGE plpgsql;
-- Funcao principal -- Fim
SELECT public.fn_pjekz_3772_executa();

DROP FUNCTION public.fn_pjekz_3772_executa();
DROP FUNCTION public.fn_pjekz_3772_tabela_nao_existe(nome_schema varchar, nome_tabela varchar);

--##################################################################################################
--##################################################################################################
-- PJE_2.3.0_074__DDL_PJEKZ-3069_CRIAR_TABELA_TB_PTE_EXTRATOR_LOG
--##################################################################################################
--##################################################################################################
/*
 * Objetivo      : Criação da tabela tb_pte_log_extracao
 * Issue         : [PJEKZ-3069]
 * SubSistema    : (X)1ºGrau (X)2ºGrau (X)3ºGrau
 * Descricao     : Criação da tabela tb_pte_log_extracao para guardar as informações sobre o procedimento de extração do módulo Pesquisa Textual
 *
 * Equipe        : CTPJE/SMPAD
 * Issue         : [PJEKZ-3069]
 * Autor         : Ana Carolina Pereira Rocha Martins
 * Email		 : acprocha@tst.jus.br
 * Data Criacao  : 12/06/2018  
 * 
*/

CREATE OR REPLACE FUNCTION public.fn_pjekz_3069_tabela_nao_existe(nome_schema varchar, nome_tabela varchar)
RETURNS boolean AS $BODY$
DECLARE
  var_table_exists smallint;

BEGIN
        SELECT 1 INTO var_table_exists
        FROM information_schema.TABLES
        WHERE table_schema = nome_schema
        AND table_name     = nome_tabela;
    
        return (var_table_exists is null);
END;
$BODY$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION public.fn_pjekz_3069_executa()
RETURNS void AS $BODY$
BEGIN

	IF  (public.fn_pjekz_3069_tabela_nao_existe('pje_pte', 'tb_pte_log_extracao')) then
	
		DROP SEQUENCE IF EXISTS  pje_pte.sq_tb_pte_log_extracao;
		CREATE SEQUENCE pje_pte.sq_tb_pte_log_extracao;
		
		ALTER TABLE pje_pte.sq_tb_pte_log_extracao OWNER TO pje;
		
		GRANT SELECT, UPDATE ON SEQUENCE pje_pte.sq_tb_pte_log_extracao TO pjero_usuario_servico_pte;
	
		/*==============================================================*/
		/* Table: tb_pte_log_extracao                                   */
		/*==============================================================*/
		create table pje_pte.tb_pte_log_extracao (
		   id_pte_log_extracao  INTEGER              not null  default nextval('pje_pte.sq_tb_pte_log_extracao'),
		   cd_tribunal          VARCHAR(30)          ,
		   dh_processamento     TIMESTAMP            not null,
		   in_sucesso           CHAR(1)              not null,
		   		constraint tb_pte_log_extracao_cc01 check (in_sucesso in ('S','N')),
		   ds_erro              TEXT                 null,
		   cd_tipo_documento    VARCHAR(13)          null,
		      	constraint tb_pte_log_extracao_cc02 check (cd_tipo_documento is null or (cd_tipo_documento in ('HTML','PDF','PDF ESCANEADO'))),
		   hh_tempo_processamento TIME               not null,
		   nm_host_processamento VARCHAR(255)        not null,
		   id_processo_documento INTEGER             not null,
		   constraint tb_pte_log_extracao_pk primary key (id_pte_log_extracao)
		);
		
		comment on table pje_pte.tb_pte_log_extracao is
		'Tabela que armazena as informações de log do extrator do módulo Pesquisa Textual';
		
		comment on column pje_pte.tb_pte_log_extracao.id_pte_log_extracao is
		'Identificador único da tabela';
		
		comment on column pje_pte.tb_pte_log_extracao.cd_tribunal is
		'Código do Tribunal de onde provem o dado';
		
		comment on column pje_pte.tb_pte_log_extracao.dh_processamento is
		'Data do processamento';
		
		comment on column pje_pte.tb_pte_log_extracao.in_sucesso is
		'Indicador se houve sucesso ou falha no procedimento de extração';
		
		comment on column pje_pte.tb_pte_log_extracao.ds_erro is
		'Descrição do erro ocorrido no procedimento de extração';
		
		comment on column pje_pte.tb_pte_log_extracao.cd_tipo_documento is
		'Tipo do documento carregado';
		
		comment on column pje_pte.tb_pte_log_extracao.hh_tempo_processamento is
		'Tempo decorrido (em segundos) no procedimento de extração';
		
		comment on column pje_pte.tb_pte_log_extracao.nm_host_processamento is
		'Nome do host de onde se desparou a extração';
		
		comment on column pje_pte.tb_pte_log_extracao.id_processo_documento is
		'Id do documento carregado';

		-- set table ownership
		alter table pje_pte.tb_pte_log_extracao owner to pje;
	
		grant DELETE,INSERT,SELECT,UPDATE on pje_pte.tb_pte_log_extracao to pjero_usuario_manutencao;

		grant DELETE,INSERT,SELECT,UPDATE on pje_pte.tb_pte_log_extracao to pjero_usuario_servico;

		/*==============================================================*/
		/* Index: tb_pte_log_extracao_in01                              */
		/*==============================================================*/
		create  index tb_pte_log_extracao_in01 on pje_pte.tb_pte_log_extracao (
		dh_processamento
		);
		
		/*==============================================================*/
		/* Index: tb_pte_log_extracao_in02                              */
		/*==============================================================*/
		create  index tb_pte_log_extracao_in02 on pje_pte.tb_pte_log_extracao (
		ds_erro
		);

	END IF;
	
END;
$BODY$ LANGUAGE plpgsql;

SELECT public.fn_pjekz_3069_executa();

DROP FUNCTION public.fn_pjekz_3069_executa();



--##################################################################################################
--##################################################################################################
-- PJE_2.3.0_151__DDL_CTPJE-8599_REPLACE_VIEW_VS_PTE_DOCUMENTO_INDEXAVEL
--##################################################################################################
--##################################################################################################
/*
 * Objetivo      : Replace da view vs_pte_documento_indexavel
 * Issue         : [CTPJE-8599]
 * SubSistema    : (X)1ºGrau (X)2ºGrau (X)3ºGrau
 * Descricao     : Replace da view vs_pte_documento_indexavel para listar apenas os documentos ativos e o nome das pessoas que assinaram separadas por virgula 
 *
 * Equipe        : CTPJE/SPTRI
 * Issue         : [CTPJE-8599]
 * Autor         : Guilherme Dantas Bispo
 * Email		 : guilherme.bispo@tst.jus.br
 * Data Criacao  : 11/10/2018
 * 
*/

CREATE OR REPLACE FUNCTION public.fn_ctpje_8599_executa()
RETURNS void AS $BODY$
BEGIN

	DROP VIEW IF EXISTS pje_pte.vs_pte_documento_indexavel;

	/*==============================================================*/
	/* View: vs_pte_documento_indexavel              */
	/*==============================================================*/

	CREATE OR REPLACE VIEW pje_pte.vs_pte_documento_indexavel AS
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
		    tpd.cd_documento as cd_tipo_processo_documento,
		    tpd.ds_tipo_processo_documento,
			(SELECT STRING_AGG(DISTINCT assin.ds_nome_pessoa,', ') 
			FROM tb_proc_doc_bin_pess_assin AS assin
			WHERE assin.id_processo_documento_bin::integer = docbin.id_processo_documento_bin::integer) as ds_nome,
		    doc.dt_juntada,
		    docbin.ds_modelo_documento as tx_conteudo_documento,
		    pte.cd_status_documento,
		    ( SELECT tribunal.cd_sigla_tribunal
		           FROM tribunal) AS cd_tribunal,
		    doc.ds_instancia AS nr_grau,
		    (SELECT ds_identificador_unico FROM tb_processo_documento doc2 WHERE doc2.id_processo_documento = doc.id_documento_principal) AS ds_identificador_unico_documento_principal,
		    to_char(nr_sequencia,'FM0000000') || '-' || to_char(nr_digito_verificador,'FM00') || '.' || to_char(nr_ano,'FM0000') || '.' || left(ptrf.nr_identificacao_orgao_justica::varchar,1) || '.' || to_char(right(ptrf.nr_identificacao_orgao_justica::varchar,2)::integer ,'FM00') || '.' || to_char(nr_origem_processo,'FM0000')  as nr_processo,
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
		     JOIN tb_processo_trf ptrf ON ptrf.id_processo_trf::integer = doc.id_processo::integer
		     JOIN tb_tipo_processo_documento tpd ON tpd.id_tipo_processo_documento::integer = doc.id_tipo_processo_documento::integer
		     LEFT JOIN tb_orgao_julgador oj ON ptrf.id_orgao_julgador::integer = oj.id_orgao_julgador::integer
		     LEFT JOIN tb_orgao_julgador_colgiado ojc ON  ojc.id_orgao_julgador_colegiado::integer = ptrf.id_orgao_julgador_colegiado::integer
		     LEFT JOIN tb_processo_assunto pa ON pa.id_processo_trf::integer = ptrf.id_processo_trf::integer
		     LEFT JOIN tb_assunto_trf a ON a.id_assunto_trf = pa.id_assunto_trf::integer
		     LEFT JOIN pje_pte.tb_pte_status_documento_indexado pte ON pte.id_processo_documento = doc.id_processo_documento
		  WHERE doc.dt_juntada IS NOT NULL 
		    AND doc.in_ativo = 'S'
		  	AND ((select grau.vl_variavel from grau) = '3' or 
				(doc.ds_instancia='3' OR doc.ds_instancia::text =(select grau.vl_variavel from grau)))
		  	AND ptrf.cd_processo_status = 'D'::bpchar
		    AND pa.in_assunto_principal = 'S' and ptrf.in_segredo_justica = 'N' and doc.in_documento_sigiloso = 'N'
			AND doc.id_tipo_processo_documento = (SELECT id_tipo_processo_documento FROM pje.tb_tipo_processo_documento WHERE ds_tipo_processo_documento = 'Recurso Ordinário');
           

	ALTER TABLE pje_pte.vs_pte_documento_indexavel
	  OWNER TO pje;
	  
	GRANT SELECT ON TABLE pje_pte.vs_pte_documento_indexavel TO pjero_usuario_servico_pte;

	COMMENT ON VIEW pje_pte.vs_pte_documento_indexavel
	  IS 'View utilizada pelo Módulo Pesquisa Textual para listar informações dos documentos não indexados';

END;
$BODY$ LANGUAGE plpgsql;

SELECT public.fn_ctpje_8599_executa();

DROP FUNCTION public.fn_ctpje_8599_executa();

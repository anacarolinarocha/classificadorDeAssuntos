# ### ====================================================================================
# Script que recupara todos os processo assunto de um regional (Segundo grau) e salva num csv
# ### ====================================================================================

from datetime import timedelta
import sys
import pandas as pd
import pandas.io.sql as psql
import time
import csv
import psycopg2
import os

sys.path.insert(1, '/home/anarocha/Documents/credentials')
from credentials import *


#SELECT QUE BUSCA OS PROCESSOS QUE FORAM REMETIDOS AO 2 GRAU E SEUS ASSUNTOS EM PRIMEIRO GRAU
sql_original = """WITH vs_nivel_assunto as(SELECT b.*,
CASE
		WHEN b.nivel = '2' THEN
			(SELECT pai.cd_assunto_trf::text FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))
		WHEN b.nivel = '3' THEN
			(SELECT avo.cd_assunto_trf::text FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf)))
		WHEN b.nivel = '4' THEN
			(SELECT bisavo.cd_assunto_trf::text FROM tb_assunto_trf bisavo WHERE bisavo.id_assunto_trf =
			(SELECT avo.id_assunto_trf_superior FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))))
		WHEN b.nivel = '5' THEN
			(SELECT tataravo.cd_assunto_trf::text FROM tb_assunto_trf tataravo WHERE tataravo.id_assunto_trf =
			(SELECT bisavo.id_assunto_trf_superior FROM tb_assunto_trf bisavo WHERE bisavo.id_assunto_trf =
			(SELECT avo.id_assunto_trf_superior FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf)))))
	ELSE b.cd_assunto_trf::text END  as cd_assunto_nivel_1,
CASE
		WHEN b.nivel = '1' THEN  ''::text
		WHEN b.nivel = '3' THEN
			(SELECT pai.cd_assunto_trf::text FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))
		WHEN b.nivel = '4' THEN
			(SELECT avo.cd_assunto_trf::text FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf)))
		WHEN b.nivel = '5' THEN
			(SELECT bisavo.cd_assunto_trf::text FROM tb_assunto_trf bisavo WHERE bisavo.id_assunto_trf =
			(SELECT avo.id_assunto_trf_superior FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))))
	ELSE b.cd_assunto_trf::text END  as cd_assunto_nivel_2,
CASE
		WHEN b.nivel = '1' THEN  ''
		WHEN b.nivel = '2' THEN  ''
		WHEN b.nivel = '4' THEN
			(SELECT pai.cd_assunto_trf::text FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))
		WHEN b.nivel = '5' THEN
			(SELECT avo.cd_assunto_trf::text FROM tb_assunto_trf avo WHERE avo.id_assunto_trf =
			(SELECT pai.id_assunto_trf_superior FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf)))
	ELSE b.cd_assunto_trf::text END  as cd_assunto_nivel_3,
CASE
		WHEN b.nivel = '1' THEN  ''
		WHEN b.nivel = '2' THEN  ''
		WHEN b.nivel = '3' THEN  ''
		WHEN b.nivel = '5' THEN
			(SELECT pai.cd_assunto_trf::text FROM tb_assunto_trf pai WHERE pai.id_assunto_trf = (SELECT aa.id_assunto_trf_superior FROM pje.tb_assunto_trf aa WHERE id_assunto_trf = b.id_assunto_trf))
	ELSE b.cd_assunto_trf::text END  as cd_assunto_nivel_4,
CASE
		WHEN b.nivel = '1' THEN  ''::text
		WHEN b.nivel = '2' THEN  ''::text
		WHEN b.nivel = '3' THEN  ''::text
		WHEN b.nivel = '4' THEN  ''::text
	ELSE b.cd_assunto_trf::text END  as cd_assunto_nivel_5
FROM (
SELECT a.*,
		CASE WHEN a.cd_assunto_trf in ('9985','899','864','6191','8826') THEN '1'::text
		WHEN a.cd_assunto_trf in ('10186','10186','9997','10370','10421','10645','10394','10088','10409','9986','10385','10954','10219','','7947','7681','10431','55054','2622','55006','7644','1654','55336','55341','55077','1695','1658','2662',
		'7628','10568','2581','2620','2567','55218','1937','1397','55078','6197','9565','6213','6218','6202','6215','6212','8893','8938','8828','9148','9192','8875','11781','8883','8842','55230','8960','9045','55301',
		'55581','55596','55619') THEN '2'::text
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
		'55604','55610','55620','55622','55627','55634') THEN '3'::text
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
		 THEN '4'::text
		WHEN a.cd_assunto_trf in ('10700','10701','10875','10893','10883','10884','10699','55016','55318','55319','55027','55028','55061','55062','55075','55331','55332','55333','55334','55335','55076','55350','55351','55110','55107','55103',
		'55130','55131','55135','55390','55391','55392','55134','55393','55136','55140','55138','55394','55395','55139','55191','55409','55410','55411','55412','55208','55413','55414','55426','55429','55430','55431',
		'55247','55435','55441','55442','55443','55444','55445','55446','55447','55448','55449','55450','55451','55452','55453','55454','55455','55272','55274','55275','55273','55461','55466','55469','55473','55477',
		'55478','55228','55493','55494','55495','55496','55497','55505','55509','55510','55511','55512','55520','55521','55522','55526','55527','55528','55531','55532','55533','55534','55535','55536','55550','55551',
		'55558','55577','55629','55630','55631','55633') THEN '5'::text
		ELSE 'Nível não identificado'::text END AS nivel
	FROM tb_assunto_trf a
	WHERE in_ativo = 'S'
) AS b) 
select p2g.nr_processo as processo_2g, a2g.cd_assunto_trf as cd_assunto_2g, 
vs.cd_assunto_nivel_1,vs.cd_assunto_nivel_2,vs.cd_assunto_nivel_3,vs.cd_assunto_nivel_4,vs.cd_assunto_nivel_5
from tb_processo_assunto pa2g
inner join pje.tb_assunto_trf a2g on a2g.id_assunto_trf = pa2g.id_assunto_trf
inner join tb_processo p2g on p2g.id_processo = pa2g.id_processo_trf
inner join tb_processo_trf ptrf on ptrf.id_processo_trf = pa2g.id_processo_trf
inner join vs_nivel_assunto vs on vs.id_assunto_trf = pa2g.id_assunto_trf
and ptrf.in_segredo_justica = 'N'
and p2g.nr_processo  is not null and p2g.nr_processo <> ''

"""
sql_original = sql_original.replace('\n', ' ')

def recuperaDadosRegional(regionais):

    for  sigla_trt in regionais:
        # sigla_trt='01'
        print("----------------------------------------------------------------------------")
        print("PROCESSANDO DADOS DE CONTINGENCIA DO TRT {} - 2o GRAU".format(sigla_trt))
        start_time = time.time()
        porta = '5' + sigla_trt + '2'
        nomeArquivo = '/media/DATA/classificadorDeAssuntos/Dados/naoPublicavel/Contingencia/TRT_' + sigla_trt  + '_2G_2010-2019_listaAssuntosProcessosNoSegundoGrauSemSegredo.csv'

        try:
            conn = psycopg2.connect(dbname=dbname_2g, user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)
            # conn = psycopg2.connect(dbname='pje_2grau_consulta', user=userbugfix, password=senhabugfix, host=ipbugfix, port=porta)

            sql_count = """select max(id_processo_assunto) from tb_processo_assunto"""
            total_registros = (psql.read_sql(sql_count,conn))
            total_registros = total_registros['max'][0]
            print('Encontrados ' + str(total_registros) + ' registros na tabela tb_processo_assunto do TRT ' + sigla_trt)
            if os.path.isfile(nomeArquivo):
               os.remove(nomeArquivo)
            chunk_size = 10000
            offset = 10000
            dfs=[]
            while True:
            # for i in range(1,5):
                sql = sql_original +" and pa2g.id_processo_assunto > %d and pa2g.id_processo_assunto < %d  limit %d " % (offset-chunk_size,offset, chunk_size)
                dfs.append(psql.read_sql(sql, conn))
                if offset == 10000 :
                    print('Primeiros dados recuperados ...' + sql[-100:])
                    dfs[-1].to_csv(nomeArquivo, mode='a', header=True, quoting=csv.QUOTE_ALL)
                else:
                    dfs[-1].to_csv(nomeArquivo, mode='a', header=False, quoting=csv.QUOTE_ALL)
                offset += chunk_size
                # if len(dfs[-1]) < chunk_size:
                #     print('Dados recuperados com sucesso.' )
                #     break;
                if offset > total_registros + chunk_size:
                    print('Ultimo sql executado ...' + sql[-100:])
                    print('Dados recuperados com sucesso.')
                    break;
            total_time = time.time() - start_time
            print('\nTempo para recuperar dados: ' + str(timedelta(seconds=(total_time))))
        except Exception as e:
            print("\033[91mNão foi possível se conectar na base do  TRT " + sigla_trt + "\033[0m")
            print(e)
            continue;
#
# for i in range (1,25):
#     recuperaDadosRegional([("{:02d}".format(i))])
recuperaDadosRegional(['08','20'])

# import multiprocessing as mp
# pool = mp.Pool(4)
# results = [pool.apply(recuperaDadosRegional, args=([("{:02d}".format(i))])) for i in range (1,25)]
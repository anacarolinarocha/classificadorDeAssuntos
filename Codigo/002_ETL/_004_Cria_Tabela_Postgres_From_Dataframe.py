#https://www.reddit.com/r/Python/comments/690j1q/faster_loading_of_dataframes_from_pandas_to/
import random
import pandas as pd
from sqlalchemy import create_engine, MetaData
import io

# df = pd.DataFrame()
# dfLen = 10000
# dfCols = 10
# for x in range(0, dfCols):
#     colName = 'a' + str(x)
#     df[colName] = [random.randint(0, 99) for x in range(1,dfLen)]

def cleanColumns(columns):
    cols = []
    for col in columns:
        col = col.replace(' ', '_')
        cols.append(col)
    return cols

def to_pg(df, table_name, stringCon):
    con = create_engine(stringCon)
    data = io.StringIO()
    df.columns = cleanColumns(df.columns)
    df.to_csv(data, header=False, index=False)
    data.seek(0)
    raw = con.raw_connection()
    curs = raw.cursor()
    curs.execute("DROP TABLE IF EXISTS " + table_name )
    empty_table = pd.io.sql.get_schema(df, table_name, con = con)
    empty_table = empty_table.replace('"', '')
    curs.execute(empty_table)
    curs.copy_from(data, table_name, sep = ',')
    curs.connection.commit()

# get_ipython().magic(u"timeit to_pg(df, 'tb_teste', stringCon)")
# get_ipython().magic(u"timeit df.to_sql(con=dw, name='tb_teste', if_exists='replace', index=False)")
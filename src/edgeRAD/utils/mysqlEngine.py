from sqlalchemy import create_engine, Table, Column, MetaData, String, BigInteger
from edgeRAD.utils.constants import *
import pandas as pd


class MysqlEngine:
    def __init__(self) -> None:
        self.mysql_engine = create_engine(
            f'mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
            # f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

        )

    def insert_batch(self, tableName, data):
        finished = False
        while not finished:
            try:
                data.to_sql(name=tableName,
                            con=self.mysql_engine,
                            if_exists='append',
                            index=False)
                finished = True
            except KeyboardInterrupt:
                exit(0)
            except:
                print("Exception occured, Retrying")

    def invoke_sql(self, sql):
        return pd.read_sql(sql, self.mysql_engine)

    def getAllServices(self, temp=False):
        return self.invoke_sql(
            "SELECT DISTINCT service_name FROM `service_state`;")

    def insert_strategy(self, data_list):
        # 创建元数据
        metadata = MetaData()
        # 定义表结构
        strategy_table = Table('strategy', metadata,
                               Column('service_name', String(255)),
                               Column('timepoint', BigInteger),
                               Column('strategy', String(511)))
        with self.mysql_engine.connect() as connection:
            sql = f"INSERT INTO strategy (service_name, timepoint, strategy) VALUES "
            sql += ", ".join([
                f"('{item['service_name']}', {item['timepoint']}, '{item['strategy']}')"
                for item in data_list
            ])
            sql += " ON DUPLICATE KEY UPDATE strategy=VALUES(strategy)"
            connection.execute(sql)

    def __del__(self):
        self.mysql_engine.dispose()

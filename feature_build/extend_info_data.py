#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Dan'
# lh copy from SloopScoreCard and modify it to use online

import json
import datetime
# from risk.data.share_data import BaseData
Extend_Info_Columns = ['device_info',  # todo text model, after getting all device_info | group 1
                       'tongdundata',  # finished
                       'tanzhidata',   # finished
                       'nifadata',     # finished
                       'app_list',     # Empty
                       'scorpionaccessreport',  # finished | A
                       'jxlaccessreport',       # finished | A
                       'qh360accessreport',     # finished | A size 7795
                       'rong360accessreport',   # not parsed. len = 3606, not A type
                       'loanbondaccessreport',  # Empty
                       'score',                 # finished
                       'hsfundjson',            # Empty
                       'tongdunguarddata',      # finished
                       'original_type',         # nothing to do
                       'emailbilljson',         # Empty
                       'emailreportjson',       # Empty
                       'providentfund51accessreport',   # Empty
                       'rongshuaccessreport',           # Empty
                       'youmengdata',                   # finished
                       'not_resolve_field',             # Empty
                       'bqsreportdata',                 # Empty
                       'datamhreportdata',              # Empty
                       'tjrong360accessreport']         # Empty


class ExtendInfoData:
    """
    read columns from table `test_20w_extend_info` and parse json
    """
    def __init__(self,is_live=False):
        if is_live:
            self.cutoff_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            # todo check cutoff_time
            # todo use test_20w_order_detail.loan_time
            self.cutoff_time = datetime.datetime.strptime('2019-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')

    def get_parsed_ei_column(self, column ,extend_data):
        if column not in Extend_Info_Columns:
            # todo add log info
            return None
        r = extend_data
        # todo, a comprehensive approach
        if column == 'device_info':
            return self.__parse_device_info(r)
        if column == 'tongdundata':
            return self.__parse_tongdundata(r)
        if column == 'tanzhidata':
            return self.__parse_tanzhidata(r)
        if column == 'nifadata':
            return self.__parse_nifadata(r)
        if column == 'scorpionaccessreport':
            return self.__parse_scorpionaccessreport(r)
        if column == 'jxlaccessreport':
            return self.__parse_jxlaccessreport(r)
        if column == 'score':
            return self.__parse_score(r)
        if column == 'tongdunguarddata':
            return self.__parse_tongdunguarddata(r)
        if column == 'youmengdata':
            return self.__parse_youmengdata(r)
        if column == 'qh360accessreport':
            return self.__parse_qh360accessreport(r)
        return None

    @staticmethod
    def __parse_device_info(r):
        if len(r) == 0:
            return None
        if r is None:
            return None
        d = r.get('device_info')
        rlt_dict = dict()
        rlt_dict['applist'] = d.get('applist') if d is not None else None
        rlt_dict['osver'] = d.get('osver') if d is not None else None
        rlt_dict['imei'] = d.get('imei') if d is not None else None
        rlt_dict['device'] = d.get('device') if d is not None else None
        return rlt_dict

    @staticmethod
    def __parse_tongdundata(r):
        if r is None or len(r) == 0:
            return None
        if r is None:
            return None
        return r.get('tongdunData')

    @staticmethod
    def __parse_tanzhidata(r):
        if r is None or len(r) == 0:
            return None
        if r is None:
            return None
        tanzhiData = r.get('tanzhiData')
        if tanzhiData:
            return tanzhiData.get('data')
        else:
            return None

    @staticmethod
    def __parse_nifadata(r):
        if r is None or len(r) == 0:
            return None
        if r is None:
            return None
        nifaData = r.get('nifaData')
        if nifaData:
            return nifaData.get('data')
        else:
            return None

    @staticmethod
    def __parse_scorpionaccessreport(r):
        if r is None or len(r) == 0:
            return None
        if r is None:
            return None
        try:
            scorpionAccessReport = r.get('scorpionAccessReport')
            if scorpionAccessReport:
                return scorpionAccessReport.get('JSON_INFO')
            else:
                return None
        except:
            return json.loads(r.get('scorpionAccessReport')).get('JSON_INFO')

    @staticmethod
    def __parse_jxlaccessreport(r):
        if r is None or len(r) == 0:
            return None
        if r is None:
            return None

        jxlAccessReport = r.get('jxlAccessReport')
        if jxlAccessReport:
            return jxlAccessReport.get('JSON_INFO')
        else:
            return None

    @staticmethod
    def __parse_score(r):
        if r is None or len(r) == 0:
            return None
        if r is None:
            return None
        return r.get('score')

    @staticmethod
    def __parse_tongdunguarddata(r):
        if r is None or len(r) == 0:
            return None
        if r is None:
            return None
        return r.get('tongdunGuardData')

    @staticmethod
    def __parse_youmengdata(r):
        if r is None or len(r) == 0:
            return None
        if r is None:
            return None
        return r.get('youmengdata')

    @staticmethod
    def __parse_qh360accessreport(r):
        if r is None or len(r) == 0:
            return None
        if r is None:
            return None
        qh360AccessReport = r.get('qh360AccessReport')
        if qh360AccessReport:
            dataJson = qh360AccessReport.get('dataJson')
            if dataJson:
                return dataJson.get('report_list')
        return None

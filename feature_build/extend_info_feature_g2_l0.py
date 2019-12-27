#!/usr/bin/python
# -*- coding: UTF-8 -*-
# lh copy from SloopScoreCard and modify it to use online
__author__ = 'Dan'

# todo add log
from feature_build.extend_info_data import ExtendInfoData
from feature_build.utils import *
import pandas as pd
import itertools
import collections
import time


class RiskFeatureGroupExtendInfoG2(ExtendInfoData):
    def __init__(self,is_live=False):
        super(RiskFeatureGroupExtendInfoG2, self).__init__(is_live)
    @staticmethod
    def __tongdundata_rulesDetail(dic):
        feat = dict()
        ls = dic.get('rules')
        if ls is None:
            return feat

        prefix = 'EI_tongdundata_rulesDetail'
        for l in ls:
            rule_id = l.get('rule_id')
            feat['{p}_ruleId{rule_id}_score'.format(p=prefix, rule_id=rule_id)] = wrapper_float(l.get('score'))
            cc = l.get('conditions')
            if cc is None or len(cc) == 0:
                continue
            for ci in cc:
                feat['{p}_ruleId{rule_id}_result'.format(p=prefix, rule_id=rule_id)] = wrapper_float(ci.get('result'))
                if 'hits' in ci.keys():
                    val = ci.get('hits')
                    for val_i in val:
                        feat['{p}_ruleId{rule_id}_hits_{idn}'.format(p=prefix, rule_id=rule_id,
                                                                     idn=val_i.get('industry_display_name'))] \
                            = wrapper_float(val_i.get('count'))
                if 'results_for_dim' in ci.keys():
                    val = ci.get('results_for_dim')
                    for val_i in val:
                        feat['{p}_ruleId{rule_id}_resultsForDim_{dim_type}'.format(p=prefix, rule_id=rule_id,
                                                                                   dim_type=val_i.get('dim_type'))] \
                            = wrapper_float(val_i.get('count'))
                if 'hits_for_dim' in ci.keys():
                    val = ci.get('hits_for_dim')
                    for val_i in val:
                        feat['{p}_ruleId{rule_id}_hitsForDim_{dim_type}_{idn}'.format(p=prefix, rule_id=rule_id,
                                                                                      dim_type=val_i.get('dim_type'),
                                                                                      idn=val_i.get('industry_display_name'))] = \
                            wrapper_float(val_i.get('count'))

        return feat

    @staticmethod
    def __tongdundata_policy_set(ls):
        feat = dict()
        if ls is None or len(ls) == 0:
            return feat

        prefix = 'EI_tongdundata_policySet'
        for val in ls:
            policy_name = val.get('policy_name')
            policy_score = wrapper_float(val.get('policy_score'))
            policy_mode = val.get('policy_mode')
            risk_type = val.get('risk_type')
            policy_decision = val.get('policy_decision')
            hit_rules = val.get('hit_rules')

            feat['{p}_pname{pname}_policyScore'.format(p=prefix, pname=policy_name)] = policy_score
            feat['{p}_pname{pname}_riskType'.format(p=prefix, pname=policy_name)] = risk_type
            feat['{p}_pname{pname}_policyMode'.format(p=prefix, pname=policy_name)] = policy_mode
            feat['{p}_pname{pname}_policyDecision'.format(p=prefix, pname=policy_name)] = policy_decision
            if hit_rules is not None:
                for hr in hit_rules:
                    feat['{p}_pname{pname}_hitRules_{name}'.format(p=prefix, pname=policy_name, name=hr.get('name'))] \
                        = wrapper_float(hr.get('score'))

        return feat

    @staticmethod
    def __tongdundata_hit_rules(ls):
        feat = dict()
        if ls is None or len(ls) == 0:
            return feat
        prefix = 'EI_tongdundata_hitRules'

        for l in ls:
            name = l.get('name')
            score = wrapper_float(l.get('score'))
            feat['{p}_hitRules_{name}_score'.format(p=prefix, name=name)] = score

        return feat

    def _feat_ei_tongdundata(self,extend_data):
        feat = dict()
        j = self.get_parsed_ei_column('tongdundata',extend_data)

        if j is None:
            return feat

        prefix = 'EI_tongdundata'

        k = 'final_score'
        feat['{p}_{k}'.format(p=prefix, k=k)] = wrapper_float(j.get(k))

        k = 'risk_type'
        feat['{p}_{k}'.format(p=prefix, k=k)] = j.get(k)

        k = 'final_decision'
        feat['{p}_{k}'.format(p=prefix, k=k)] = j.get(k)

        feat.update(self.__tongdundata_rulesDetail(j.get('rulesDetail')))
        feat.update(self.__tongdundata_policy_set(j.get('policy_set')))
        feat.update(self.__tongdundata_hit_rules(j.get('hit_rules')))

        return feat

    @staticmethod
    def __tanzhidata_feat0(j_data):
        feat = dict()
        credit_info_keys = ['refInfos', 'platform_Infos', 'eveSums']
        for cik in credit_info_keys:
            ls = j_data.get('mb_infos')[0].get('credit_info').get(cik)
            for l in ls:
                slice_name = l.get('slice_name')
                ls_keys = l.keys()
                ls_keys = [vv for vv in ls_keys if vv != 'slice_name']
                for key in ls_keys:
                    feat['EI_tanzhidata_{cik}_{slice_name}_{key}'.format(slice_name=slice_name, key=key,
                                                                         cik=cik)] = l.get(key)

        return feat

    @staticmethod
    def __tanzhidata_feat1(j_data):
        feat = dict()
        ls = j_data.get('mb_infos')[0].get('credit_info').get('sections')
        sections_columns = ['section_name', 'apply_request_count', 'repay_fail_count', 'overdue_repay_maxdelay_level',
                            'verif_count', 'overdue_repay_average_level',
                            'overdue_average_level', 'repay_remind_average_level',
                            'overdue_count', 'register_count', 'overdue_repay_count',
                            'apply_reject_count', 'apply_request_average_level',
                            'repay_fail_average_level', 'overdue_maxdelay_level',
                            'repay_remind_count', 'loan_offer_count',
                            'loan_offer_average_level']
        df_section_name = pd.DataFrame(columns=sections_columns)
        for l in ls:
            # todo section name to a abstract time window
            ls_vals = []
            for col in sections_columns:
                ls_vals.append(l.get(col))
            df_section_name.loc[len(df_section_name)] = ls_vals
        # replace '' with 0
        df_section_name = df_section_name.replace('', None)
        df_section_name = df_section_name.sort_values(by=['section_name'], ascending=0)
        df_section_name = df_section_name.iloc[0:50]
        df_section_name.index = range(50)
        del sections_columns[0]
        for col in sections_columns:
            for k in range(50):
                feat['EI_tanzhidata_mb_infos_lastweek{k}_{col}'.format(k=k, col=col)] = df_section_name[col].iloc[k]

        return feat

    def _feat_ei_tanzhidata(self, extend_data):
        feat = dict()
        j_data = self.get_parsed_ei_column( 'tanzhidata',extend_data)
        if j_data is None:
            return feat

        feat.update(self.__tanzhidata_feat0(j_data))
        feat.update(self.__tanzhidata_feat1(j_data))
        feat = wrapper_missing_value(feat)

        # Note that *_level feature is category: ['A' to 'E']
        return feat

    def _feat_ei_nifadata(self,extend_data):
        feat = dict()
        j = self.get_parsed_ei_column( 'nifadata',extend_data)
        if j is None:
            return feat

        ls_keys = ['overduemoreamt',
                   'loancount',
                   'loanbal',
                   'queryatotalorg',
                   'outstandcount',
                   'overdueamt',
                   'loanamt',
                   'generationcount',
                   'generationamount',
                   'overduemorecount',
                   'totalorg']
        for key in ls_keys:
            feat['EI_nifadata_{key}'.format(key=key)] = j.get(key)

        # print(np.unique(feat.values()))
        return feat

    @staticmethod
    def __scorpionaccessreport_behavior(ls):
        feat = dict()
        keys = ['cell_operator_zh', 'net_flow', 'call_out_time', 'cell_operator', 'call_in_cnt', 'cell_phone_num',
                'sms_cnt', 'cell_loc', 'call_cnt', 'total_amount', 'call_out_cnt', 'call_in_time']
        df0 = pd.DataFrame(columns=keys + ['cell_mth'])
        for l in ls:
            ls_val = []
            for k in keys:
                ls_val.append(l.get(k))
            ls_val.append(l.get('cell_mth'))
            df0.loc[len(df0)] = ls_val

        df0 = df0.sort_values(by=['cell_mth'], ascending=0)
        df0 = df0.iloc[0:6]
        df0.index = range(len(df0))

        for col in keys:
            for k in range(len(df0)):
                feat['EI_sar_JSON_INFO_lastmth{k}_{col}'.format(k=k, col=col)] = df0[col].iloc[k]
        return feat

    @staticmethod
    def __scorpionaccessreport_contact_region(ls):
        feat = dict()
        keys = ['region_avg_call_out_time', 'region_call_in_time_pct', 'region_call_out_cnt_pct',
                'region_call_out_time_pct', 'region_call_in_time', 'region_avg_call_in_time', 'region_call_in_cnt_pct',
                'region_call_out_time', 'region_call_out_cnt', 'region_call_in_cnt', 'region_uniq_num_cnt']
        for l in ls:
            region_loc = l.get('region_loc')
            for k in keys:
                feat['EI_sar_region{region_loc}_{k}'.format(region_loc=region_loc, k=k)] = l.get(k)
        return feat

    @staticmethod
    def __scorpionaccessreport_appcheck(ls, cutoff_time):
        prefix = 'EI_sar_appcheck'
        # todo clean data, dummy or text model; this function only returns raw data
        feat = dict()

        ls1 = ls[1].get('check_points')
        feat['{p}_ls1_court_blacklist'.format(p=prefix)] = ls1.get('court_blacklist').get('arised')
        feat['{p}_ls1_financial_blacklist'.format(p=prefix)] = ls1.get('financial_blacklist').get('arised')

        ls2 = ls[2].get('check_points')
        feat['{p}_ls2_financial_blacklist'.format(p=prefix)] = ls2.get('financial_blacklist').get('arised')

        feat['{p}_ls2_website'.format(p=prefix)] = ls2.get('website')
        feat['{p}_ls2_check_idcard'.format(p=prefix)] = ls2.get('check_idcard')
        feat['{p}_ls2_reliability'.format(p=prefix)] = ls2.get('reliability')

        reg_time = wrapper_strptime(ls2.get('reg_time'))
        diff_time = (cutoff_time - reg_time).days/365 if reg_time is not None else missingValue
        feat['{p}_ls2_diff_reg_time'.format(p=prefix)] = diff_time

        feat['{p}_ls2_check_name'.format(p=prefix)] = ls2.get('check_name')
        feat['{p}_ls2_webcheck_ebusinesssite'.format(p=prefix)] = ls2.get('webcheck_ebusinesssite')

        # 运营商通讯地址
        ls3 = ls[3]
        feat['{p}_ls3_operator_addr'.format(p=prefix)] = ls3.get('check_points').get('key_value')
        return feat

    @staticmethod
    def __scorpionaccessreport_trip_info(ls):
        feat = dict()
        df0 = pd.DataFrame(columns=['trip_type', 'trip_dest', 'trip_leave', 'trip_start_time', 'trip_end_time'])
        for l in ls:
            df0.loc[len(df0)] = [l.get('trip_type'), l.get('trip_dest'), l.get('trip_leave'), l.get('trip_start_time'),
                                 l.get('trip_end_time')]

        df0.trip_start_time = [wrapper_strptime(k) for k in df0.trip_start_time]
        df0.trip_end_time = [wrapper_strptime(k) for k in df0.trip_end_time]
        df0 = df0.assign(trip_diff_day=wrapper_diff_time(df0, 'trip_start_time', 'trip_end_time', 'hour'))

        # todo gps `trip_dest`, `trip_leave`

        prefix = 'EI_sar_tripInfo'
        feat['{p}_cnt'.format(p=prefix)] = len(df0)
        uni_trip_types = [u'双休日', u'节假日', u'工作日']
        for trip_type in uni_trip_types:
            feat['{p}_cnt_{trip_type}'.format(p=prefix, trip_type=trip_type)] = sum(
                df0['trip_type'] == trip_type)
            feat['{p}_Ratio_{trip_type}'.format(p=prefix, trip_type=trip_type)] = \
                wrapper_div(feat.get('{p}_cnt_{trip_type}'.format(p=prefix, trip_type=trip_type)), len(df0))

        feat['{p}_cnt_unique_trip_dest'.format(p=prefix)] = len(np.unique(df0['trip_dest']))
        feat['{p}_cnt_unique_trip_leave'.format(p=prefix)] = len(np.unique(df0['trip_leave']))

        # metrics for df0.trip_diff_day
        feat.update(apply_basic_metrics(list(df0.trip_diff_day), '{p}_trip_diff_time_'.format(p=prefix)))
        return feat

    @staticmethod
    def __scorpionaccessreport_main_service(ls):
        feat = dict()
        prefix = 'EI_sar_main_service'

        # convert ls to df
        # todo call convert_ls_to_df
        df = pd.DataFrame(columns=['total_service_cnt', 'company_type', 'company_name'])
        for l in ls:
            df.loc[len(df)] = [l.get('total_service_cnt'), l.get('company_type'), l.get('company_name')]

        feat.update(apply_basic_metrics(list(df.total_service_cnt), '{p}_totalServiveCnt'.format(p=prefix)))

        feat['{p}_cnt_unique_company_type'.format(p=prefix)] = len(np.unique(df.company_type))

        uni_company_types = np.unique(df.company_type)
        for ctype in uni_company_types:
            feat['{p}_cnt_company_type_{ctype}'.format(p=prefix, ctype=ctype)] = sum(df.company_type == ctype)
            feat['{p}_ratio_company_type_{ctype}'.format(p=prefix, ctype=ctype)] = \
                wrapper_div(sum(df.company_type == ctype), len(df))
        return feat

    @staticmethod
    def __scorpionaccessreport_contact_list(ls):

        feat = dict()
        prefix = 'EI_sar_contact_list'

        cols = ['contact_name', 'needs_type',
                'contact_all_day', 'contact_early_morning', 'contact_morning', 'contact_noon',
                'contact_afternoon', 'contact_night',
                'call_in_len', 'call_out_len', 'call_len',
                'call_in_cnt', 'call_out_cnt', 'call_cnt',
                'contact_1w', 'contact_1m', 'contact_3m', 'contact_3m_plus',
                'phone_num', 'phone_num_loc', 'p_relation',
                'contact_weekday', 'contact_weekend',  'contact_holiday']
        # df = convert_ls_to_df(ls, cols)
        df = convert_ls_to_df_better(ls, cols)

        # df = pd.concat([cols, df], join='inner')


        # basic info `contact_name`
        # TODO TODO apply a sophisticated phone classifier or topic model to `contact_name`

        c = collections.Counter(df.needs_type)
        for k, v in c.items():
            feat['{p}_cnt_needsType_{k}'.format(p=prefix, k=k)] = v
            feat['{p}_ratioOfCnt_needsType_{k}'.format(p=prefix, k=k)] = wrapper_div(v, len(df))

        c = collections.Counter(df.contact_all_day)
        for k, v in c.items():
            feat['{p}_cnt_contactAllDay_{k}'.format(p=prefix, k=str(k))] = v
            feat['{p}_ratioOfCnt_contactAllDay_{k}'.format(p=prefix, k=str(k))] = wrapper_div(v, len(df))

        c = collections.Counter(df.phone_num_loc)
        for k, v in c.items():
            feat['{p}_cnt_phoneNumLoc_{k}'.format(p=prefix, k=str(k))] = v
            feat['{p}_ratioOfCnt_phoneNumLoc_{k}'.format(p=prefix, k=str(k))] = wrapper_div(v, len(df))

        cols0 = ['contact_early_morning', 'contact_morning', 'contact_noon',
                 'contact_afternoon', 'contact_night']
        cols1 = ['call_in_len', 'call_out_len', 'call_len', 'call_in_cnt', 'call_out_cnt', 'call_cnt']
        cols2 = ['contact_1w', 'contact_1m', 'contact_3m', 'contact_3m_plus']
        cols3 = ['contact_weekday', 'contact_weekend', 'contact_holiday']
        for col in cols0 + cols1 + cols2 + cols3:
            feat.update(apply_basic_metrics(list(df[col]), '{p}_{col}'.format(p=prefix, col=col)))

        # apply emd metric, df-not-filtered
        for cols in [cols0, cols1, cols2, cols3]:
            for win_tup in itertools.combinations(cols, 2):
                feat['{p}_EMD_{c1}_{c2}'.format(p=prefix, c1=win_tup[0], c2=win_tup[1])] = \
                    wrapper_emd(df[win_tup[0]], df[win_tup[1]])

        # TODO TODO apply time window as filters
        return feat

    @staticmethod
    def __scorpionaccessreport_behavior_check(ls):
        feat = dict()
        prefix = 'EI_sar_behaviorCheck'
        for l in ls:
            k = l.get('check_point')
            v = l.get('score')
            feat['{p}_{k}_score'.format(p=prefix, k=k)] = v
        return feat

    @staticmethod
    def __scorpionaccessreport_user_info_check(ls):
        feat = dict()
        prefix = 'EI_sar_userInfoCheck'

        ls0 = ls.get('check_search_info')
        ls1 = ls.get('check_black_info')

        ls0_keys = ['phone_with_other_idcards', 'phone_with_other_names', 'register_org_cnt', 'arised_open_web',
                    'searched_org_cnt', 'idcard_with_other_phones', 'searched_org_type', 'register_org_type',
                    'idcard_with_other_names']
        for k in ls0_keys:
            v = ls0.get(k)
            v = len(v) if isinstance(v, list) else v
            feat['{p}_{k}'.format(p=prefix, k=k)] = v

        ls1_keys = ['contacts_class1_cnt', 'contacts_class1_blacklist_cnt', 'contacts_class2_blacklist_cnt',
                    'contacts_router_cnt', 'contacts_router_ratio', 'phone_gray_score']
        for k in ls1_keys:
            v = ls1.get(k)
            v = 0 if v is None else v
            feat['{p}_{k}'.format(p=prefix, k=k)] = v
        return feat

    def _feat_ei_scorpionaccessreport(self,extend_data):
        feat = dict()
        j = self.get_parsed_ei_column( 'scorpionaccessreport',extend_data)
        # similar structure as scorpion access report, so map to scorpion features
        if j is None:
            j = self.get_parsed_ei_column('jxlaccessreport',extend_data)
        if j is None:
            j = self.get_parsed_ei_column( 'qh360accessreport',extend_data)

        if j is None:
            return feat

        # print(j.get('deliver_address'))    # empty
        # print(j.get('collection_contact')) # empty
        # j.get('ebusiness_expense') # empty


        feat.update(self.__scorpionaccessreport_behavior(ls=j.get('cell_behavior')[0].get('behavior')))
        # feat.update(self.__scorpionaccessreport_contact_region(ls=j.get('contact_region')))
        feat.update(self.__scorpionaccessreport_appcheck(ls=j.get('application_check'), cutoff_time=self.cutoff_time))
        feat.update(self.__scorpionaccessreport_trip_info(ls=j.get('trip_info')))
        feat.update(self.__scorpionaccessreport_main_service(ls=j.get('main_service')))
        feat.update(self.__scorpionaccessreport_contact_list(ls=j.get('contact_list')))  # Todo speedup
        feat.update(self.__scorpionaccessreport_behavior_check(ls=j.get('behavior_check')))
        feat.update(self.__scorpionaccessreport_user_info_check(ls=j.get('user_info_check')))

        return feat

    def _feat_ei_score(self,extend_data):
        feat = dict()
        prefix = 'EI_score'
        j = self.get_parsed_ei_column( 'score',extend_data)
        if j is None:
            return feat

        for l in j:
            score_val = l.get('score')
            name = l.get('name')
            if isinstance(score_val, str):
                score_val = missingValue if len(score_val) == 0 else score_val
            feat['{p}_{name}'.format(p=prefix, name=name)] = score_val

        return feat

    def _feat_ei_tongdunguarddata(self, extend_data):
        feat = dict()
        prefix = 'EI_tongdunguarddata'
        j = self.get_parsed_ei_column( 'tongdunguarddata',extend_data)
        if j is None:
            return feat

        ks = ['rEazz31000011', 'rBazz01132010', 'rGcaz11000030',
              'rEbzz39000011', 'rChzz03038030', 'rAbzz03030010', 'isHistory',
              'rAcbz03009011', 'rEbzz25000020', 'rAbzz03101041', 'rEbzz21000011']
        for k in ks:
            v = self.__process_tongdungarddata(j.get(k))
            feat['{p}_{k}_val0'.format(p=prefix, k=k)] = v[0]
            feat['{p}_{k}_val1'.format(p=prefix, k=k)] = v[1]

        return feat

    @staticmethod
    def __process_tongdungarddata(val):
        v1 = missingValue
        v2 = missingValue
        if val:
            val = val.replace('[', '').replace(')', '')
            val = val.split(',')
            if len(val) == 2:
                v1 = wrapper_float(val[0])
                v2 = wrapper_float(val[1])
        return [v1, v2]

    def _feat_ei_youmengdata(self, extend_data):
        feat = dict()
        j = self.get_parsed_ei_column( 'youmengdata',extend_data)
        if j is None:
            return feat

        prefix = 'EI_youmengdata'
        pre_keys = ['car', 'sns',  'finance', 'top', 'tail', 'appStability', 'property', 'travel', 'entertainment',
                    'service', 'education', 'woman', 'reading', 'tools', 'shopping', 'game', 'loan', 'health']

        for d in ['180d', '90d', '7d']:
            for pk in pre_keys:
                key_val = pk+d
                feat['{p}_{k}'.format(p=prefix, k=key_val)] = j.get(key_val)
        return feat

    def calc_the_group(self, extend_data):
        # join with feature json and if not calculated, replace by missing value.

        starttime = time.time()

        all_feat_dict = dict()
        all_feat_dict.update(self._feat_ei_tongdundata(extend_data))
        # all_feat_dict.update(self._feat_ei_tanzhidata(extend_data))
        all_feat_dict.update(self._feat_ei_nifadata(extend_data))
        all_feat_dict.update(self._feat_ei_scorpionaccessreport(extend_data))
        all_feat_dict.update(self._feat_ei_score(extend_data))
        all_feat_dict.update(self._feat_ei_tongdunguarddata(extend_data))
        # all_feat_dict.update(self._feat_ei_youmengdata(extend_data))

        endtime = time.time()
        print(' cost time: ', endtime - starttime)

        return all_feat_dict

# coding=utf-8
import os
import numpy as np
import csv
import sed_eval


class EvaluationBasic(object):
    def __init__(self):
        pass

    def delimiter(self, filename):
        sniffer = csv.Sniffer()
        valid_delimiters = ['\t', ',', ';', ' ']
        delimiter = '\t'
        with open(filename, 'rt') as f1:
            try:
                example_content = f1.read(1024)
                dialect = sniffer.sniff(example_content)
                if hasattr(dialect, '_delimiter'):
                    if dialect._delimiter in valid_delimiters:
                        delimiter = dialect._delimiter
                elif hasattr(dialect, 'delimiter'):
                    if dialect.delimiter in valid_delimiters:
                        delimiter = dialect.delimiter
                else:
                    # Fall back to default
                    delimiter = '\t'
            except:
                # Fall back to default
                delimiter = '\t'
        return delimiter

    def is_number(self, string):
        try:
            float(string)  # for int, long and float
        except ValueError:
            try:
                complex(string)  # for complex
            except ValueError:
                return False
        return True

    def load_result(self, file):
        data = []
        with open(file, 'rt') as f:
            audio = file.split('/')[-1].split('.')[0] + '.wav'
            # print audio
            for row in csv.reader(f, delimiter=self.delimiter(file)):
                if row:
                    row_format = []
                    for item in row:
                        row_format.append(self.is_number(item))  # 检测每一项是否是数字
                    # print row_format  # ['0.16', '3.02', 'car']  [True, True, False]
                    if len(row) == 3 and row_format == [True, True, False]:
                        # Format: [event_onset  event_offset    event_label]
                        data.append(({'file': audio, 'event_onset': float(row[0]),
                                      'event_offset': float(row[1]), 'event_label': row[2]}))
        return data

    def load_meta(self, file_list):
        meta_reference = []
        meta_estimate = []
        for file_pair in file_list:
            reference_meta = self.load_result(file_pair['reference'])  # 读取每一个文件
            estimate_meta = self.load_result(file_pair['estimate'])  # 分别读取没一个文件
            # 因为评估只能一个文件一个文件的进行，所以还是用append吧
            meta_reference.append(reference_meta)  # list.append(object) 向列表中添加一个对象object
            meta_estimate.append(estimate_meta)  # list.extend(sequence) 把一个序列seq的内容添加到列表中
        # print len(meta_reference)
        # print meta_reference
        # print len(meta_estimate)
        # print meta_estimate
        return [meta_reference, meta_estimate]

    def output(self, overall_metrics_per_scene, hyb_detial=False):
        output = ''
        # output += " \n"
        output += "  Overall metrics \n"
        output += "  =============== \n"  # 都是一个一个的累加，所以是 +=
        output += "    {event_label:<17s} | {segment_based_fscore:7s} | {segment_based_er:7s} | {event_based_fscore:7s} | {event_based_er:7s} | " \
                  "\n".format(event_label='Event label',
                              segment_based_fscore='Seg. F1',
                              segment_based_er='Seg. ER',
                              event_based_fscore='Evt. F1',
                              event_based_er='Evt. ER', )
        # Event label       | Seg. F1 | Seg. ER | Evt. F1 | Evt. ER |
        # output += "    {event_label:<17s} + {segment_based_fscore:7s} + {segment_based_er:7s} + {event_based_fscore:7s} + {event_based_er:7s} " \
        #           "+ \n".format(event_label='-' * 17,
        #                         segment_based_fscore='-' * 7,
        #                         segment_based_er='-' * 7,
        #                         event_based_fscore='-' * 7,
        #                         event_based_er='-' * 7, )
        #  ----------------- + ------- + ------- + ------- + ------- +
        avg = {'segment_based_fscore': [],
               'segment_based_er': [],
               'event_based_fscore': [],
               'event_based_er': [], }
        # scene_label = 'street'
        # output += "    {scene_label:<17s} | {segment_based_fscore:<7s} | {segment_based_er:<7s} | {event_based_fscore:<7s} | {event_based_er:<7s} | " \
        #           "\n".format(scene_label=scene_label,
        #                       segment_based_fscore="{:4.2f}".format(overall_metrics_per_scene.get_path(
        #                           'segment_based_metrics.overall.f_measure.f_measure') * 100),
        #                       segment_based_er="{:4.2f}".format(
        #                           overall_metrics_per_scene.get_path(
        #                               'segment_based_metrics.overall.error_rate.error_rate')),
        #                       event_based_fscore="{:4.2f}".format(
        #                           overall_metrics_per_scene.get_path(
        #                               'event_based_metrics.overall.f_measure.f_measure') * 100),
        #                       event_based_er="{:4.2f}".format(
        #                           overall_metrics_per_scene.get_path(
        #                               'event_based_metrics.overall.error_rate.error_rate')), )
        # #  street            | 40.84   | 0.90    | 2.67    | 3.02    |
        # # print avg['segment_based_fscore']
        avg['segment_based_fscore'].append(
            overall_metrics_per_scene.get_path('segment_based_metrics.overall.f_measure.f_measure') * 100)
        # print(avg['segment_based_fscore'])  # [32.61363636363636]
        # 最后的scene的F值就是这里的平均F值，但是他们是怎么算出来的？
        avg['segment_based_er'].append(
            overall_metrics_per_scene.get_path('segment_based_metrics.overall.error_rate.error_rate'))
        avg['event_based_fscore'].append(
            overall_metrics_per_scene.get_path('event_based_metrics.overall.f_measure.f_measure') * 100)
        avg['event_based_er'].append(
            overall_metrics_per_scene.get_path('event_based_metrics.overall.error_rate.error_rate'))

        output += "    {scene_label:<17s} + {segment_based_fscore:7s} + {segment_based_er:7s} + {event_based_fscore:7s} + {event_based_er:7s} + \n".format(
            scene_label='-' * 17,
            segment_based_fscore='-' * 7,
            segment_based_er='-' * 7,
            event_based_fscore='-' * 7,
            event_based_er='-' * 7,
        )
        #     ----------------- + ------- + ------- + ------- + ------- +
        # print("avg['segment_based_fscore']:", avg['segment_based_fscore'])
        # print("avg['segment_based_er']:", avg['segment_based_er'])
        # print("avg['event_based_fscore']:", avg['event_based_fscore'])
        # print("avg['event_based_er']:", avg['event_based_er'])

        output += "    {scene_label:<17s} | {segment_based_fscore:<7s} | {segment_based_er:<7s} | {event_based_fscore:<7s} | {event_based_er:<7s} | \n".format(
            scene_label='Average',
            segment_based_fscore="{:4.2f}".format(np.mean(avg['segment_based_fscore'])),
            segment_based_er="{:4.2f}".format(np.mean(avg['segment_based_er'])),
            event_based_fscore="{:4.2f}".format(np.mean(avg['event_based_fscore'])),
            event_based_er="{:4.2f}".format(np.mean(avg['event_based_er'])),
        )
        #     Average           | 40.84   | 0.90    | 2.67    | 3.02    |

        if hyb_detial:
            print('output:\n', output)
        # print(output)

        return avg['segment_based_fscore'][0], avg['segment_based_er'][0], avg['event_based_fscore'][0], avg['event_based_er'][0]


class ContainerMixin(object):  # 这个类的作用全在评估中体现
    def get_path(self, dotted_path, default=None, data=None):
        """Get value from nested dict with dotted path 通过点路径从嵌套的dict获取值
        Parameters
        ----------
        dotted_path : str 点路：str
            String in form of "field1.field2.field3"  “field1.field2.field3”形式的字符串
        default : str, int, float
            Default value returned if path does not exists  如果路径不存在，返回默认值
            Default value "None"  默认值“无”
        data : dict, optional
            Dict for which path search is done, if None given self is used. Used for recursive path search.
            Default value "None" 对于哪个路径搜索完成，如果没有给定自我使用。 用于递归路径搜索。 默认值“无”
        Returns
        -------
        """
        if data is None: #如果数据为空
            data = self #模型等于本身
        fields = dotted_path.split('.') #领域列表 = 切分

        if '*' == fields[0]: #如果 * 是列表第一个元素
            # Magic field to return all childes in a list 魔术字段返回列表中的所有孩子
            sub_list = []
            from six import iteritems
            for key, value in iteritems(data): #条目数据中的所有键、值
                if len(fields) > 1:
                    sub_list.append(self.get_path(data=value, dotted_path='.'.join(fields[1:]), default=default))
                else:
                    sub_list.append(value)
            return sub_list
        else: #如果不是
            if fields[0] in data and len(fields) > 1:
                # Go deeper 更深一点
                return self.get_path(data=data[fields[0]], dotted_path='.'.join(fields[1:]), default=default)
            elif fields[0] in data and len(fields) == 1:
                # We reached to the node 到了节点
                return data[fields[0]]
            else:
                return default

    def _walk(self, d, depth=0):
        """Recursive dict walk to get string of the content nicely formatted

        Parameters
        ----------
        d : dict
            Dict for walking
        depth : int
            Depth of walk, string is indented with this
            Default value 0

        Returns
        -------
            str

        """

        output = ''
        indent = 3
        header_width = 35 - depth*indent

        for k, v in sorted(d.items(), key=lambda x: x[0]):
            if isinstance(v, dict):
                output += "".ljust(depth * indent)+k+'\n'
                output += self._walk(v, depth + 1)
            else:
                if isinstance(v, np.ndarray):
                    # np array or matrix
                    shape = v.shape
                    if len(shape) == 1:
                        output += "".ljust(depth * indent)
                        output += k.ljust(header_width) + " : " + "array (%d)" % (v.shape[0]) + '\n'

                    elif len(shape) == 2:
                        output += "".ljust(depth * indent)
                        output += k.ljust(header_width) + " : " + "matrix (%d,%d)" % (v.shape[0], v.shape[1]) + '\n'

                elif isinstance(v, list) and len(v) and isinstance(v[0], str):
                    output += "".ljust(depth * indent) + k.ljust(header_width) + " : list (%d)\n" % len(v)
                    for item_id, item in enumerate(v):
                        output += "".ljust((depth + 1) * indent)
                        output += ("["+str(item_id)+"]").ljust(header_width-3) + " : " + str(item) + '\n'

                elif isinstance(v, list) and len(v) and isinstance(v[0], np.ndarray):
                    # List of arrays
                    output += "".ljust(depth * indent) + k.ljust(header_width) + " : list (%d)\n" % len(v)
                    for item_id, item in enumerate(v):
                        if len(item.shape) == 1:
                            output += "".ljust((depth+1) * indent)
                            output += ("["+str(item_id)+"]").ljust(header_width-3) + " : array (%d)" % (item.shape[0]) + '\n'

                        elif len(item.shape) == 2:
                            output += "".ljust((depth+1) * indent)
                            output += ("["+str(item_id)+"]").ljust(header_width-3) + " : matrix (%d,%d)" % (item.shape[0], item.shape[1]) + '\n'

                elif isinstance(v, list) and len(v) and isinstance(v[0], dict):
                    output += "".ljust(depth * indent)
                    output += k.ljust(header_width) + " : list (%d)\n" % len(v)

                    for item_id, item in enumerate(v):
                        output += "".ljust((depth + 1) * indent) + "["+str(item_id)+"]" + '\n'
                        output += self._walk(item, depth + 2)

                else:
                    output += "".ljust(depth * indent) + k.ljust(header_width) + " : " + str(v) + '\n'

        return output

    def __str__(self):
        return self._walk(self, depth=1)

    def show(self):
        """Print container content

        Returns
        -------
            Nothing

        """

        print(self._walk(self, depth=1))

    @staticmethod
    def _search_list_of_dictionaries(key, value, list_of_dictionaries):
        """Search in the list of dictionaries

        Parameters
        ----------
        key : str
            Dict key for the search
        value :
            Value for the key to match
        list_of_dictionaries : list of dicts
            List to search

        Returns
        -------
            Dict or None

        """

        for element in list_of_dictionaries:
            if element.get(key) == value:
                return element
        return None

    def merge(self, override, target=None):
        """ Recursive dict merge

        Parameters
        ----------
        target : dict
            target parameter dict

        override : dict
            override parameter dict

        Returns
        -------
        None

        """

        if not target:
            target = self
        from six import iteritems
        for k, v in iteritems(override):
            if k in target and isinstance(target[k], dict) and isinstance(override[k], dict):
                self.merge(target=target[k], override=override[k])
            else:
                target[k] = override[k]

    def _clean_for_hashing(self, data, non_hashable_fields=None):
        # Recursively remove keys with value set to False, or non hashable fields
        if non_hashable_fields is None and hasattr(self, 'non_hashable_fields'):
            non_hashable_fields = self.non_hashable_fields
        elif non_hashable_fields is None:
            non_hashable_fields = []

        if data:
            if 'enable' in data and not data['enable']:
                return {
                    'enable': False,
                }
            else:
                if isinstance(data, dict):
                    for key in list(data.keys()):
                        value = data[key]
                        if isinstance(value, bool) and value is False:
                            # Remove fields marked False
                            del data[key]
                        elif key in non_hashable_fields:
                            # Remove fields marked in non_hashable_fields list
                            del data[key]
                        elif isinstance(value, dict):
                            if 'enable' in value and not value['enable']:
                                # Remove dict block which is disabled
                                del data[key]
                            else:
                                # Proceed recursively
                                data[key] = self._clean_for_hashing(value)
                    return data
                else:
                    return data
        else:
            return data

class DottedDict(dict, ContainerMixin):  # 看来这个类还得放在前面
    def __init__(self, *args, **kwargs):
        super(DottedDict, self).__init__(*args, **kwargs)

        self.non_hashable_fields = [
            '_hash',
            'verbose',
        ]
        if kwargs.get('non_hashable_fields'):
            self.non_hashable_fields.update(kwargs.get('non_hashable_fields'))

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.__dict__ = state


def sound_event_detection(reference_path, estimate_path, label_list, detail=None, testing_filenames_361=None):
    estimate_files = os.listdir(estimate_path)
    file_list = []
    for file in estimate_files:
        # Mu - Too Bright.txt
        file_id = file.replace('.txt', '')
        if testing_filenames_361:
            if file_id in testing_filenames_361:
                # print('file_id:', file_id)
                dic = {}
                dic['reference'] = os.path.join(reference_path, file_id + '.txt')
                dic['estimate'] = os.path.join(estimate_path, file_id + '.txt')
                file_list.append(dic)
        else:
            dic = {}
            dic['reference'] = os.path.join(reference_path, file_id + '.txt')
            dic['estimate'] = os.path.join(estimate_path, file_id + '.txt')
            file_list.append(dic)
    # start evaluating
    overall_metrics_per_scene = {}
    # creat metrics classes, define parameters
    # print label_list
    # print type(label_list)

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=label_list, time_resolution=1.0)
    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=label_list,
        evaluate_onset=True,
        evaluate_offset=False,
        t_collar=0.5,
        percentage_of_length=0.5)  # 事件基本度量 初始化

    meta = EvaluationBasic().load_meta(file_list)
    for index in range(len(meta[0])):  # 对结果的评估只能一个文件一个文件的来
        segment_based_metric.evaluate(reference_event_list=meta[0][index], estimated_event_list=meta[1][index])
        event_based_metric.evaluate(reference_event_list=meta[0][index], estimated_event_list=meta[1][index])
        # print segment_based_metric
        # print event_based_metric

    overall_metrics_per_scene['segment_based_metrics'] = segment_based_metric.results()
    overall_metrics_per_scene['event_based_metrics'] = event_based_metric.results()
    # print(overall_metrics_per_scene['segment_based_metrics'])
    # print(overall_metrics_per_scene['event_based_metrics'])

    overall_metrics_per_scene = DottedDict(overall_metrics_per_scene)
    if detail:
        print('overall_metrics_per_scene:\n', overall_metrics_per_scene)

    # print overall_metrics_per_scene  # 就已经是没一项详细的结果，但是没有最终的总结结果
    seg_f1, seg_er, evt_f1, evt_er = EvaluationBasic().output(overall_metrics_per_scene, hyb_detial=detail)

    return seg_f1, seg_er, evt_f1, evt_er


def sound_event_detection_single_audio(current_path, reference_path, estimate_path, label_list):
    estimate_files = os.listdir(estimate_path)
    metrics_matrix = []  # 所有音频的4个结果矩阵
    audio_name_list = []
    file_list = []
    for file in estimate_files:
        each_metric_list = []
        # Mu - Too Bright.txt
        file_id = file.replace('.txt', '')
        audio_name_list.append(file_id)
        dic = {}
        dic['reference'] = os.path.join(reference_path, file_id + '_meta.txt')
        dic['estimate'] = os.path.join(estimate_path, file_id + '.txt')
        file_list.append(dic)

        # start evaluating
        overall_metrics_per_scene = {}
        # creat metrics classes, define parameters
        # print label_list
        # print type(label_list)

        segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
            event_label_list=label_list, time_resolution=1.0)
        event_based_metric = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=label_list,
            evaluate_onset=True,
            evaluate_offset=False,
            t_collar=0.5,
            percentage_of_length=0.5)  # 事件基本度量 初始化

        meta = EvaluationBasic().load_meta(file_list)
        for index in range(len(meta[0])):  # 对结果的评估只能一个文件一个文件的来
            segment_based_metric.evaluate(reference_event_list=meta[0][index], estimated_event_list=meta[1][index])
            event_based_metric.evaluate(reference_event_list=meta[0][index], estimated_event_list=meta[1][index])
            # print segment_based_metric
            # print event_based_metric

        overall_metrics_per_scene['segment_based_metrics'] = segment_based_metric.results()
        overall_metrics_per_scene['event_based_metrics'] = event_based_metric.results()
        # print(overall_metrics_per_scene['segment_based_metrics'])
        # print(overall_metrics_per_scene['event_based_metrics'])

        overall_metrics_per_scene = DottedDict(overall_metrics_per_scene)
        # print('overall_metrics_per_scene:\n', overall_metrics_per_scene)
        # overall_metrics_per_scene:
        #     event_based_metrics
        #       class_wise
        #          accompaniment
        #             accuracy
        #             count
        #                Nref                 : 32.0
        #                Nsys                 : 24.0
        #             error_rate
        #                deletion_rate        : 0.4375
        #                error_rate           : 0.625
        #                insertion_rate       : 0.1875
        #             f_measure
        #                f_measure            : 0.6428571428571429
        #                precision            : 0.75
        #                recall               : 0.5625
        #          silence
        #             accuracy
        #             count
        #                Nref                 : 31.0
        #                Nsys                 : 18.0
        #             error_rate
        #                deletion_rate        : 0.5161290322580645
        #                error_rate           : 0.6129032258064516
        #                insertion_rate       : 0.0967741935483871
        #             f_measure
        #                f_measure            : 0.6122448979591837
        #                precision            : 0.8333333333333334
        #                recall               : 0.4838709677419355
        #          vocals
        #             accuracy
        #             count
        #                Nref                 : 62.0
        #                Nsys                 : 171.0
        #             error_rate
        #                deletion_rate        : 0.2903225806451613
        #                error_rate           : 2.338709677419355
        #                insertion_rate       : 2.0483870967741935
        #             f_measure
        #                f_measure            : 0.37768240343347637
        #                precision            : 0.2573099415204678
        #                recall               : 0.7096774193548387
        #       class_wise_average
        #          accuracy
        #          error_rate
        #             deletion_rate           : 0.41465053763440857
        #             error_rate              : 1.1922043010752688
        #             insertion_rate          : 0.7775537634408601
        #          f_measure
        #             f_measure               : 0.544261481416601
        #             precision               : 0.6135477582846004
        #             recall                  : 0.5853494623655914
        #       overall
        #          accuracy
        #          error_rate
        #             deletion_rate           : 0.368
        #             error_rate              : 1.456
        #             insertion_rate          : 1.072
        #             substitution_rate       : 0.016
        #          f_measure
        #             f_measure               : 0.4556213017751479
        #             precision               : 0.3615023474178404
        #             recall                  : 0.616
        #    segment_based_metrics
        #       class_wise
        #          accompaniment
        #             accuracy
        #                accuracy             : 0.9851632047477745
        #                balanced_accuracy    : 0.8214285714285714
        #                sensitivity          : 1.0
        #                specificity          : 0.6428571428571429
        #             count
        #                Nref                 : 323.0
        #                Nsys                 : 328.0
        #             error_rate
        #                deletion_rate        : 0.0
        #                error_rate           : 0.015479876160990712
        #                insertion_rate       : 0.015479876160990712
        #             f_measure
        #                f_measure            : 0.9923195084485407
        #                precision            : 0.9847560975609756
        #                recall               : 1.0
        #          silence
        #             accuracy
        #                accuracy             : 0.9643916913946587
        #                balanced_accuracy    : 0.8615315315315315
        #                sensitivity          : 0.7297297297297297
        #                specificity          : 0.9933333333333333
        #             count
        #                Nref                 : 37.0
        #                Nsys                 : 29.0
        #             error_rate
        #                deletion_rate        : 0.2702702702702703
        #                error_rate           : 0.32432432432432434
        #                insertion_rate       : 0.05405405405405406
        #             f_measure
        #                f_measure            : 0.8181818181818181
        #                precision            : 0.9310344827586207
        #                recall               : 0.7297297297297297
        #          vocals
        #             accuracy
        #                accuracy             : 0.8189910979228486
        #                balanced_accuracy    : 0.8104367325931916
        #                sensitivity          : 0.8522167487684729
        #                specificity          : 0.7686567164179104
        #             count
        #                Nref                 : 203.0
        #                Nsys                 : 204.0
        #             error_rate
        #                deletion_rate        : 0.1477832512315271
        #                error_rate           : 0.30049261083743845
        #                insertion_rate       : 0.15270935960591134
        #             f_measure
        #                f_measure            : 0.85012285012285
        #                precision            : 0.8480392156862745
        #                recall               : 0.8522167487684729
        #       class_wise_average
        #          accuracy
        #             accuracy                : 0.9228486646884274
        #             balanced_accuracy       : 0.8311322785177649
        #             sensitivity             : 0.8606488261660675
        #             specificity             : 0.8016157308694623
        #          error_rate
        #             deletion_rate           : 0.13935117383393247
        #             error_rate              : 0.21343227044091784
        #             insertion_rate          : 0.07408109660698538
        #          f_measure
        #             f_measure               : 0.886874725584403
        #             precision               : 0.9212765986686237
        #             recall                  : 0.8606488261660675
        #       overall
        #          accuracy
        #             accuracy                : 0.9228486646884273
        #             balanced_accuracy       : 0.9220653070286728
        #             sensitivity             : 0.9289520426287744
        #             specificity             : 0.9151785714285714
        #          error_rate
        #             deletion_rate           : 0.07104795737122557
        #             error_rate              : 0.13854351687388988
        #             insertion_rate          : 0.0674955595026643
        #             substitution_rate       : 0.0
        #          f_measure
        #             f_measure               : 0.9306049822064056
        #             precision               : 0.9322638146167558
        #             recall                  : 0.9289520426287744

        # print overall_metrics_per_scene  # 就已经是没一项详细的结果，但是没有最终的总结结果
        # EvaluationBasic().output(overall_metrics_per_scene)

        output = ''

        seg_f = overall_metrics_per_scene.get_path('segment_based_metrics.overall.f_measure.f_measure') * 100
        seg_er = overall_metrics_per_scene.get_path('segment_based_metrics.overall.error_rate.error_rate')
        event_f = overall_metrics_per_scene.get_path('event_based_metrics.overall.f_measure.f_measure') * 100
        event_er = overall_metrics_per_scene.get_path('event_based_metrics.overall.error_rate.error_rate')
        # print(seg_f, seg_er, event_f, event_er)
        # segment_based_fscore="{:4.2f}".format(np.mean(avg['segment_based_fscore'])),
        #             segment_based_er="{:4.2f}".format(np.mean(avg['segment_based_er'])),
        #             event_based_fscore="{:4.2f}".format(np.mean(avg['event_based_fscore'])),
        #             event_based_er="{:4.2f}".format(np.mean(avg['event_based_er']))
        each_metric_list.append(seg_f)
        each_metric_list.append(seg_er)
        each_metric_list.append(event_f)
        each_metric_list.append(event_er)
        metrics_matrix.append(each_metric_list)

    # print(audio_name_list)
    # print(len(audio_name_list))
    # print(np.array(metrics_matrix).shape)

    file_path = os.path.join(current_path, 'system/results', 'each_audio_sed_total.txt')
    with open(file_path, 'w') as file:
        for i in range(len(audio_name_list)):
            file.write(audio_name_list[i] + '\t' + str(metrics_matrix[i][0])
                       + '\t' + str(metrics_matrix[i][1])
                       + '\t' + str(metrics_matrix[i][2])
                       + '\t' + str(metrics_matrix[i][3]) + '\r\n')
            file.flush()

    file_path = os.path.join(current_path, 'system/results', 'each_audio_sed_audio_name_list.txt')
    with open(file_path, 'w') as file:
        for i in range(len(audio_name_list)):
            file.write(audio_name_list[i] + '\r\n')
            file.flush()

    file_path = os.path.join(current_path, 'system/results', 'each_audio_sed_matrix.txt')
    np.savetxt(file_path, np.array(metrics_matrix), fmt='%.4f')

    return


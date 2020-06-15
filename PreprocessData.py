# -*- coding:utf-8 -*-
import numpy as np
import os


class datasets():
    def __init__(self, name):
        self.hour_unit = 1
        self.day_unit = 24
        self.week_unit = 168
        self.PEMS_1Hour_fragment = 12
        self.pre_day = 12 if "pems08" in name else 9
        print(self.pre_day)

    def PEMS_SlideWindow(self, filename, WindowSize, PredictSize):
        '''
        Get SlideWindow Dataset
        :param filename: string, original dataset
        :param WindowSize: int, num of looking back
        :param PredictSize:int, num of predicting target
        :return:
        Sample: np.ndarray 3-dimension
        Label:np.ndarray 3-dimension
        max_data:float
        '''
        data_seq = np.load(filename)['data'][:, :, 0]
        max_data = np.max(data_seq)

        # normalization
        normal_data_seq = data_seq / max_data

        # look back past hours
        seq_len = self.hour_unit * self.PEMS_1Hour_fragment * WindowSize
        pre_len = self.hour_unit * self.PEMS_1Hour_fragment * PredictSize

        Sample = []
        Label = []
        for i in range(len(normal_data_seq) - seq_len - pre_len):
            batch_sample = normal_data_seq[i:seq_len + pre_len + i]
            Sample.append(batch_sample[0:seq_len])
            Label.append(batch_sample[seq_len: seq_len + pre_len])

        return np.array(Sample), np.array(Label), max_data

    def PEMS_adjacency_matrix(self, distance_adj_filename, nodes):
        ''' Get adjacency matrix '''
        import csv

        A = np.zeros((int(nodes), int(nodes)), dtype=np.float32)

        distaneA = np.zeros((int(nodes), int(nodes)), dtype=np.float32)

        with open(distance_adj_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distaneA[i, j] = distance
        return A, distaneA

    def PEMS_MultiComponent(self, filename,
                            num_of_weeks, num_of_days,
                            num_of_hours, num_for_predict, num_of_shift, save=True):
        '''
        Parameter
        :param filename: string, original dataset
        :param num_of_weeks: int
        :param num_of_days: int
        :param num_of_hours: int
        :param num_for_predict: int, num of predicting target
        :param num_of_shift:int, num of periodic temporal shift
        :param save:
        :return: all_sample: list
        '''
        def search_data(sequence_length, num_of_depend, label_start_idx,
                        num_for_predict, units, points_per_hour, Is_WeekOrDay, num_of_shift):
            '''
            Parameter
            :param sequence_length: int, length of history data
            :param num_of_depend: int
            :param label_start_idx: int, the first index of predicting target
            :param num_for_predict: int, num of predicting target
            :param units: int, hour:1 day:24 week:7 * 24
            :param points_per_hour: int, num of points per hour, in this dataset, it is 12
            :param Is_WeekOrDay: bool. temporal shifts only take place daily and weekly
            :param num_of_shift: int, num of periodic temporal shift
            :return: list[(start_idx, end_idx)]
            '''

            if points_per_hour < 0:
                raise ValueError("points_per_hour should be greater than 0!")

            shift_hour = num_of_shift if Is_WeekOrDay else 0

            if label_start_idx + num_for_predict > sequence_length:
                return None

            x_idx = []
            for i in range(1, num_of_depend + 1):
                start_idx = label_start_idx - points_per_hour * units * i - shift_hour * num_for_predict
                end_idx = start_idx + num_for_predict * (2 * shift_hour + 1)
                if start_idx >= 0:
                    x_idx.append((start_idx, end_idx))
                else:
                    return None

            if len(x_idx) != num_of_depend:
                return None

            return x_idx[::-1]

        def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                               label_start_idx, num_for_predict, points_per_hour=12):
            '''
            Parameter
            :param data_sequence: np.ndarray, shape is (sequence_length, nodes, features)
            :param num_of_weeks: int
            :param num_of_days: int
            :param num_of_hours: int
            :param label_start_idx: int, the first index of predicting target
            :param num_for_predict: int, num of predicting target
            :param points_per_hour: int, default is 12
            :return:
            week_sample: np.ndarray, shape is (num_of_weeks * points_per_hour, nodes, features)
            day_sample; np.ndarray, shape is (num_of_days * points_per_hour, nodes, features)
            hour_sample: np.ndarray, shape is (num_of_hours * points_per_hour, nodes, features)
            target: np.ndarray, shape is (num_for_predict, nodes, features)
            '''

            week_sample, day_sample, hour_sample = None, None, None

            if label_start_idx + num_for_predict > data_sequence.shape[0]:
                return week_sample, day_sample, hour_sample, None

            if num_of_weeks > 0:
                week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                           label_start_idx, num_for_predict,
                                           self.week_unit, points_per_hour, True, num_of_shift)
                if not week_indices:
                    return None, None, None, None

                week_sample = np.concatenate([data_sequence[i: j]
                                              for i, j in week_indices], axis=0)

            if num_of_days > 0:
                day_indices = search_data(data_sequence.shape[0], num_of_days,
                                          label_start_idx, num_for_predict,
                                          self.day_unit, points_per_hour, True, num_of_shift)
                if not day_indices:
                    return None, None, None, None

                day_sample = np.concatenate([data_sequence[i: j]
                                             for i, j in day_indices], axis=0)

            if num_of_hours > 0:
                hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                           label_start_idx, num_for_predict,
                                           self.hour_unit, points_per_hour, False, 0)
                if not hour_indices:
                    return None, None, None, None

                hour_sample = np.concatenate([data_sequence[i: j]
                                              for i, j in hour_indices], axis=0)

            target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

            return week_sample, day_sample, hour_sample, target

        data_seq = np.load(filename)['data']
        max_data = np.max(data_seq[:, :, 0])
        all_samples = []
        for idx in range(data_seq.shape[0]):
            sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        self.PEMS_1Hour_fragment)

            if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
                continue

            week_sample, day_sample, hour_sample, target = sample

            sample = []  # [(week_sample),(day_sample),(hour_sample),target]

            if num_of_weeks > 0:
                week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
                sample.append(week_sample)

            if num_of_days > 0:
                day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
                sample.append(day_sample)

            if num_of_hours > 0:
                hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
                sample.append(hour_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
            sample.append(target)

            all_samples.append(sample)

        print("the length of all_sample is {}".format(len(all_samples)))
        split_line1 = int(len(all_samples) * 0.6)

        # the pre_day of PEMSD4 and PEMSD8 is different
        split_line2 = int(len(all_samples) - 12 * 24 * self.pre_day)

        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]  # [(B,N,Tws),(B,N,Tds),(B,N,Th),(B,N,Tpre)]
        validation_set = [np.concatenate(i, axis=0)
                          for i in zip(*all_samples[split_line1: split_line2])]
        testing_set = [np.concatenate(i, axis=0)
                       for i in zip(*all_samples[split_line2:])]

        train_x = np.concatenate(training_set[:-1], axis=-1)  # (B,N,T')
        val_x = np.concatenate(validation_set[:-1], axis=-1)
        test_x = np.concatenate(testing_set[:-1], axis=-1)

        train_target = training_set[-1]  # (B,N,T)
        val_target = validation_set[-1]
        test_target = testing_set[-1]

        all_data = {
            'train': {
                'x': train_x / max_data,
                'target': train_target / max_data,
            },
            'val': {
                'x': val_x / max_data,
                'target': val_target / max_data,
            },
            'test': {
                'x': test_x / max_data,
                'target': test_target / max_data,
            },
            'max': {
                '_max': max_data,
            }
        }
        print('train x:', all_data['train']['x'].shape)
        print('train target:', all_data['train']['target'].shape)
        print()
        print('val x:', all_data['val']['x'].shape)
        print('val target:', all_data['val']['target'].shape)
        print()
        print('test x:', all_data['test']['x'].shape)
        print('test target:', all_data['test']['target'].shape)
        print()
        print('data _max :', all_data['max']['_max'])

        if save:
            files = os.path.basename(filename).split('.')[0]
            dirpath = os.path.dirname(filename)
            filename = os.path.join(dirpath,
                                    files + '_h' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(
                                        num_of_weeks) + '_p' + str(num_for_predict) + '_s' + str(
                                        num_of_shift) + '_MultiComponent')
            print('save files:', filename)
            np.savez_compressed(filename,
                                train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                                val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                                test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                                max_data=all_data['max']['_max'],
                                )

        return all_samples

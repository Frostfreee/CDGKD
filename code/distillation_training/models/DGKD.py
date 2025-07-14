import tensorflow as tf
from tensorflow.keras import Model
import random

class DGKD(Model): # Densely Guided Knowledge Distillation using Multiple Teacher Assistants
    def __init__(self, student_model=None, teacher_models=None, train_mode='teacher-TA_1', T=2, alpha=1, droprate=0.5, **kwargs):
        super(DGKD, self).__init__(**kwargs)
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.train_mode = train_mode

        self.T = T
        self.alpha = alpha
        self.droprate = droprate

        self.clf_loss = None
        self.kd_loss = None
        self.clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
        self.kd_loss_tracker = tf.keras.metrics.Mean(name='kd_loss')
        self.sum_loss_tracker = tf.keras.metrics.Mean(name='loss')

    def compile(self, clf_loss=None, kd_loss=None, **kwargs):
        super(DGKD, self).compile(**kwargs)
        self.clf_loss = clf_loss
        self.kd_loss = kd_loss

    @property
    def metrics(self):
        metrics = [self.sum_loss_tracker, self.clf_loss_tracker, self.kd_loss_tracker]

        if self.compiled_metrics is not None:
            metrics += self.compiled_metrics.metrics

        return metrics

    def train_step(self, data):
        x, y = data
        video_data, eeg_data = x

        random.seed()  

        with tf.GradientTape() as tape:
            if self.train_mode.endswith('student'):
                student_cls, student_logits = self.student_model(video_data)
            else:
                student_cls, student_logits = self.student_model(x)
            clf_loss_value = self.clf_loss(y, student_cls)

            # 计算多个蒸馏损失
            kd_losses = []   
            for teacher in self.teacher_models:
                if len(kd_losses) == 0: 
                    teacher_cls, _, _, teacher_logits, _ = teacher(x, training=False)
                else:
                    teacher_cls, teacher_logits = teacher(x, training=False)

                kd_loss_value = self.kd_loss(tf.math.softmax(student_logits/self.T), tf.math.softmax(teacher_logits/self.T))

                kd_losses.append(kd_loss_value)

            # 通过概率判断是否丢弃某个KD损失
            retained_losses = []
            for kd_loss in kd_losses:
                if random.uniform(0, 1) > self.droprate:  # 按丢弃概率判断是否保留
                    retained_losses.append(kd_loss)

            # 如果没有任何保留的蒸馏损失，则设为0
            if retained_losses:
                total_kd_loss = tf.reduce_sum(retained_losses)
            else:
                total_kd_loss = 0.0


            sum_loss_value = self.alpha * clf_loss_value + (1-self.alpha) * total_kd_loss

        self.optimizer.minimize(sum_loss_value, self.student_model.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, student_cls)

        self.sum_loss_tracker.update_state(sum_loss_value)
        self.clf_loss_tracker.update_state(clf_loss_value)
        self.kd_loss_tracker.update_state(kd_loss_value)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        video_data, eeg_data = x

        if self.train_mode.endswith('student'):
            student_cls, student_logits = self.student_model(video_data)
        else:
            student_cls, student_logits = self.student_model(x)
        clf_loss_value = self.clf_loss(y, student_cls)

        # 计算多个蒸馏损失
        kd_losses = []   
        for teacher in self.teacher_models:
            if len(kd_losses) == 0: 
                teacher_cls, _, _, teacher_logits, _ = teacher(x, training=False)
            else:
                teacher_cls, teacher_logits = teacher(x, training=False)

            kd_loss_value = self.kd_loss(tf.math.softmax(student_logits/self.T), tf.math.softmax(teacher_logits/self.T))

            kd_losses.append(kd_loss_value)

        # 通过概率判断是否丢弃某个KD损失
        retained_losses = []
        for kd_loss in kd_losses:
            if random.uniform(0, 1) > self.droprate:  # 按丢弃概率判断是否保留
                retained_losses.append(kd_loss)

        # 如果没有任何保留的蒸馏损失，则设为0
        if retained_losses:
            total_kd_loss = tf.reduce_sum(retained_losses)
        else:
            total_kd_loss = 0.0

        sum_loss_value = self.alpha * clf_loss_value + (1-self.alpha) * total_kd_loss

        self.compiled_metrics.update_state(y, student_cls)

        self.sum_loss_tracker.update_state(sum_loss_value)
        self.clf_loss_tracker.update_state(clf_loss_value)
        self.kd_loss_tracker.update_state(kd_loss_value)

        return {m.name: m.result() for m in self.metrics}

class SaveBestStudentModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, monitor='val_acc', mode='max'):
        super(SaveBestStudentModelCallback, self).__init__()
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best = -float('inf') if mode == 'max' else float('inf')  # 初始化最佳值
        self.monitor_op = tf.math.greater if mode == 'max' else tf.math.less  # 比较操作符

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # 如果当前的 val_acc 比之前最佳的要好，保存学生模型
        if self.monitor_op(current, self.best):
            self.best = current
            print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.4f}, saving student model to {self.save_path}")
            self.model.student_model.save(self.save_path)
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, model,pos_weight, norm,lambd,base):
        preds_sub = preds
        labels_sub = labels
        self.base = base
        self.model = model
        self.lambd = lambd
        if self.base == True:
            self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight)) + self.calculate_group_loss()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

            self.opt_op = self.optimizer.minimize(self.cost)
            self.grads_vars = self.optimizer.compute_gradients(self.cost)
        elif self.base == False:
            self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight)) + self.calculate_loss()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
            
            self.opt_op = self.optimizer.minimize(self.cost)
            self.grads_vars = self.optimizer.compute_gradients(self.cost)
        
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    
    def calculate_group_loss(self):
        weight_loss = 0
        name_list = ['weights0','weights1','weights2']

        for i in range(3):
            upper_column_loss = tf.norm(self.model.weight_matrix_hidden1[name_list[i]],axis = 0) # specify which layer
            loss_upper = tf.reduce_mean(upper_column_loss)
            weight_loss = weight_loss + self.lambd * loss_upper
        for i in range(3):
            bottom_column_loss = tf.norm(self.model.weight_matrix_embeddings[name_list[i]],axis = 0)
            loss_bottom = tf.reduce_mean(bottom_column_loss)
            weight_loss = weight_loss + self.lambd * loss_bottom
        return weight_loss

    def calculate_loss(self):
        weight_loss = 0
        name_list = ['weights0','weights1','weights2']

        for i in range(3):
            loss_upper = tf.norm(self.model.weight_matrix_hidden1[name_list[i]])
            weight_loss = weight_loss + self.lambd * loss_upper
        for i in range(3):
            loss_bottom = tf.norm(self.model.weight_matrix_embeddings[name_list[i]])
            weight_loss = weight_loss + self.lambd * loss_bottom
        return weight_loss



class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

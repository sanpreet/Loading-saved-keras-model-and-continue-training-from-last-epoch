import os
import tensorflow as tf
# w1 = tf.Variable(tf.random_normal(shape = [2], name = 'w1'))
# w2 = tf.Variable(tf.random_normal(shape = [5], name = 'w2'))
# # object saver to save the model files
# saver = tf.train.Saver()
# checkpoint_folder = 'tensorflow_checkpoints'
# # running the sess
# sess  = tf.Session()
# sess.run(tf.global_variables_initializer())
# for step in range(1000):
#     # saving the model in the saver
#     if step % 100 == 0:
#         saver.save(sess, os.path.join(checkpoint_folder, 'model'), global_step = step)

# restoring the model
checking_dir  = 'tensorflow_checkpoints'
with tf.Session() as sess:
  for step in range(1000):  
      if step % 100 == 0:
            # print(step)            
            try:
                print("Parameters for the step {}".format(step))
                print()
                # contains all the variables and operations
                new_saver = tf.train.import_meta_graph(checking_dir + '/' + 'model-{}.meta'.format(step))
                ckpt = tf.train.get_checkpoint_state(checking_dir)
                # print(ckpt)
                # print("Model checkpoint path is:", ckpt.model_checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    print("loading parameters ")
                    new_saver.restore(sess, ckpt.model_checkpoint_path)
                    print(sess.run('w1:0'))
                    print(sess.run('w2:0'))
                    print("------------------------------------------")
            except:
                print("Step is not valid")        
                print("---------------------------------------------")
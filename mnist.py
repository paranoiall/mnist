from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, redirect, url_for, send_from_directory
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image, ImageFilter
from werkzeug import secure_filename
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import time

global fileurl

UPLOAD_FOLDER = '/mnist/image'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

cluster = Cluster(contact_points=['cassandra'],port=9042)
session = cluster.connect()

KEYSPACE = "mykeyspace"


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()


def imageprepare():
    global fileurl
    im = Image.open(fileurl) #28*28像素
    plt.show()
    im = im.convert('L')
    tv = list(im.getdata())
    tva = [(255-x)*1.0/255.0 for x in tv] #白底黑字
    return tva

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in set(['png', 'jpg', 'jpeg'])

def createKeySpace():
    try:
        session.execute("""
            CREATE KEYSPACE IF NOT EXISTS %s
            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '1' }
        """ % KEYSPACE)
        session.set_keyspace(KEYSPACE)
        session.execute("""
            CREATE TABLE IF NOT EXISTS ytytable (
            date text,
            file text,
            result text,
            PRIMARY KEY (date)
            )
        """)
        log.info("Table created!")
    except Exception as e:
        log.error("Unable to create table!")
        log.error(e)

createKeySpace();

def insertdata(result):
    global fileurl
    try:
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        session.execute("""
            INSERT INTO ytytable (date, file, result)
            VALUES ('%s', '%s', '%s')
            """ % (now, fileurl, result)
            )
        log.info("%s, %s, %s" % (now, fileurl, result))
        log.info("Data stored!")
    except Exception as e:
        log.error("Unable to insert data!")
        log.error(e)


@app.route('/', methods=['GET','POST'])
def index():
    global fileurl
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fileurl = url_for('uploaded_file',filename=filename)
            image()
    else return """Please use ' curl 0.0.0.0:4000 -F "file=@file_url" ' """

@app.route('/html', methods=['GET','POST'])
def index_html():
    global fileurl
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fileurl = url_for('uploaded_file',filename=filename)
            return redirect('/action')
    return '''
    <!doctype html>
    <html>
    <head>
        <meta charset='UTF-8'>
        <title>Mnist</title>
    </head>
    <body>
        <br>
        <form method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
        <br>
        <input type='submit' value='submit'>
        </form>
    </body>
    </html>
    '''

@app.route('/mnist/image/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/action')
def image():
    result = imageprepare()
    saver.restore(sess, "/mnist/model/model.ckpt") #使用模型
    prediction = tf.argmax(y_conv,1)
    predint = prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
    result = str(predint[0])
    insertdata(result)
    return result

@app.route('/train')
def train():
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    saver.save(sess, '/mnist/model/model.ckpt') #储存模型


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

# mnist
首先利用docker下载cassandra的image，然后运行docker build。（可以提前下载tensorflow/tensorflow:latest-py3）<br>
>docker run --name yty-cassandra -p 9042:9042 -d cassandra<br>
>docker run --link yty-cassandra:cassandra -p 4000:80 mnist<br>
>curl 0.0.0.0:4000 -F "file=@图片url"<br><br>

可以运行cqlsh查看数据<br>
>docker run -it --link yty-cassandra:cassandra --rm cassandra cqlsh cassandra<br>
>describe keyspaces;<br>
>use mykeyspace;<br>
>select * from ytytable;<br>

![](https://github.com/paranoiall/mnist/blob/master/demo.gif)<br>
demo<br>
>https://github.com/paranoiall/mnist/blob/master/mnist.mp4<br>

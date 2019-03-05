# mnist
These codes implement the basic functionality of mnist in docker. After inputting a digital photo with a pixel of 28*28 through curl or browser, tensorflow will recognize and return the recognition result, and the test accuracy rate can reach 99.7%.
The following are specific methods of use.<br>

First use docker to download cassandra's image, then run docker build.（You can download tensorflow/tensorflow:latest-py3 in advance）<br>
>command:<br>
>>docker run --name yty-cassandra -p 9042:9042 -d cassandra<br>
>>docker run --link yty-cassandra:cassandra -p 4000:80 mnist<br>
>>curl 0.0.0.0:4000 -F "file=@url"<br><br>

Can run cqlsh to view data<br>
>command:<br>
>>docker run -it --link yty-cassandra:cassandra --rm cassandra cqlsh cassandra<br>
>>describe keyspaces;<br>
>>use mykeyspace;<br>
>>select * from ytytable;<br>

Here's a demo.<br>
<img width="320" height="180" src="https://github.com/paranoiall/mnist/blob/master/demo.gif"/><br>
Demo's url:<br>
https://github.com/paranoiall/mnist/blob/master/mnist.mp4<br>

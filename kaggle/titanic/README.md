训练集包含项
* PassengerId: 乘客编号(丢弃)
* Survived: 是否幸存
* Pclass: 乘客级别((1 = 1st; 2 = 2nd; 3 = 3rd)(地位越高, 生存几率越大)
* Name: 名字(丢弃)
* Sex: 性别(Female生存几率大)
* Age: 年龄(年纪小和大的生存几率大)
* SibSp: 船上兄弟姐妹/配偶的人数(人数越多生存几率越小)
* Parch: 船上父母/子女的人数(人数越多生存几率越小)
* Ticket: 船票号码(丢弃)
* Fare: 旅客票价(票价越高生存几率越大)
* Cabin: 船舱
* Embarked: 上船港口(C = Cherbourg; Q = Queenstown; S = Southampton)(丢弃)

train.csv例子:

```
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)",female,14,1,0,237736,30.0708,,C
11,1,3,"Sandstrom, Miss. Marguerite Rut",female,4,1,1,PP 9549,16.7,G6,S
```

train.csv例子:

```
PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
892,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q
893,3,"Wilkes, Mrs. James (Ellen Needs)",female,47,1,0,363272,7,,S
894,2,"Myles, Mr. Thomas Francis",male,62,0,0,240276,9.6875,,Q
```

# 类的初始化函数

除了构造函数和析构函数，类常用的特殊成员函数包括：
1. 拷贝构造函数：ClassName(const ClassName& other);
2. 拷贝赋值运算符：ClassName& operator=(const ClassName& other);
3. 移动构造函数：ClassName(ClassName&& other);
4. 移动赋值运算符：ClassName& operator=(ClassName&& other);
5. 友元函数：允许外部函数访问私有和保护成员。
6. 友元类：允许其他类访问私有和保护成员。
7. 转换函数：将类对象转换为其他类型。
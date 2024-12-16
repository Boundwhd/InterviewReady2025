# include<memory>
# include<string>
using namespace std;

class Character {
public:
    Character(const string& name);
    ~Character();
    void displayInfo() const;
private:
    string name_;
    shared_ptr<int> health_;
};
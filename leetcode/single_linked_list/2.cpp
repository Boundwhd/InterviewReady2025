/*两数相加*/

#include <iostream>
#include <stdio.h>

using namespace std;

struct ListNode{
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2){
    ListNode* dummy = new ListNode(0);
    ListNode* cur = dummy;
    int sum = 0;
    while(l1 || l2 || sum){
        if(l1){
            sum += l1->val;
            l1 = l1->next;
        }
        if(l2){
            sum += l2->val;
            l2 = l2->next;
        }
        cur->next = new ListNode(sum % 10);
        cur = cur->next;
        sum /= 10;
    }
    return dummy->next;
}

void printList(ListNode *head){
    while(head){
        cout << head->val << "->";
        head = head->next;
    }
    cout << "NULL" << endl;
}

int main(){
    ListNode* l1 = new ListNode(1);
    ListNode* l1_cur = l1;
    l1_cur->next = new ListNode(3);
    l1_cur = l1_cur->next;
    l1_cur->next = new ListNode(2);
    l1_cur = l1_cur->next;

    ListNode* l2 = new ListNode(6);
    ListNode* l2_cur = l2;
    l2_cur->next = new ListNode(2);
    l2_cur = l2_cur->next;
    l2_cur->next = new ListNode(1);
    l2_cur = l2_cur->next;

    // 1-3-2 + 6-2-1 = 7-5-3
    ListNode* ans = addTwoNumbers(l1, l2);
    printList(ans);
    return 0;
}
/*反转链表*/

#include <iostream>
#include <stdio.h>

using namespace std;

struct ListNode{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseList(ListNode* head) {
    if(!head) return nullptr;
    ListNode *pre = nullptr;
    ListNode *cur = head;
    while(cur){
        ListNode *next = cur->next;
        cur->next = pre;
        pre = cur;
        if(!next) break;
        cur = next;
    }
    return cur;
}

void printList(ListNode *head){
    while(head){
        cout << head->val << "->";
        head = head->next;
    }
    cout << "NULL" << endl;
}

int main(){
    ListNode *head = new ListNode(0);
    ListNode *cur = head;
    for(int i = 1;i < 6;i++){
        cur->next = new ListNode(i);
        cur = cur->next;
    }
    printList(head);
    ListNode *ans = reverseList(head);
    printList(ans);
    return 0;
}
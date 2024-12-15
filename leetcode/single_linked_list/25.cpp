/*k个一组翻转链表*/

#include <iostream>
#include <stdio.h>

using namespace std;

struct ListNode{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};


ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode* cur_nums = head;
    int node_nums = 0;
    while(cur_nums){
        cur_nums = cur_nums->next;
        node_nums++;
    }
    int cnt = node_nums / k;  

    ListNode *dummy = new ListNode(0);
    dummy->next = head;

    ListNode *pre = dummy;
    ListNode *cur = head;
    while(cnt--){
        for(int i = 0;i < k - 1;i++){
            ListNode *next = cur->next;
            cur->next = next->next;
            next->next = pre->next;
            pre->next = next;
        }
        pre = cur;
        cur = cur->next;
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
    ListNode *head = new ListNode(0);
    ListNode *cur = head;
    for(int i = 1;i < 6;i++){
        cur->next = new ListNode(i);
        cur = cur->next;
    }
    printList(head);
    ListNode *ans = reverseKGroup(head, 3);
    printList(ans);
    return 0;
}
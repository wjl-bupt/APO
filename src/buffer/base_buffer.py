# -*- encoding: utf-8 -*-
'''
@File       :base_buffer.py
@Description:
@Date       :2025/03/26 16:29:03
@Author     :junweiluo
@Version    :python
'''

class BaseBuffer(object):
    def __init__(self, args):

        self.args = args
    
    def push(self):
        raise NotImplementedError()
    
    def pop(self):
        raise NotImplementedError()


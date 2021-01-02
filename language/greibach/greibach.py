"""
实现任意文法转换为 Greibach 范式
构造下推自动机
检验输入的语句是否能够满足 Greibach 模式

@Author 黄伟
"""

from typing import List, Dict, Tuple

# 规则存储文件路径
path_pattern = r"pattern.txt"


class pattern:
    """
    规则类，每一个规则类包含以下属性：

    1.规则的开始符 name          \n
    2.规则推导出的字符列表 derive
    """

    def __init__(self, raw=""):
        self.name = ""
        self.solved_problem = ""
        if raw != "":
            self.derive = self.process(raw)  # type: List[str]

    def process(self, raw) -> List[str]:
        """
        处理输入的字符串，将其切割为推导结果列表    \n
        :param raw: 输入的字符串
        :return: 推导结果的列表
        """
        temp = raw.split("->")
        self.name = temp[0]
        return [x.replace("\n", "") for x in temp[1].split("|")]


def get_patterns() -> Dict[str, pattern]:
    """
    读取规则文件，创建规则字典
    :return: 规则字典，键为规则的开始符号，值为该规则派生字符列表
    """
    patterns = {}
    with open(path_pattern) as f:
        line = f.readline()
        while line:
            p = pattern(line)  # 新建一个规则类
            patterns[p.name] = p
            line = f.readline()
    return patterns


class greibach:
    """
    实现greibach范式的类
    """

    def __init__(self):
        self.patterns = get_patterns()  # type: Dict[str, pattern]
        self.__add_s1_pattern()

        self.empty = "Empty"
        self.orders, self.idx = self.__get_orders()  # 返回开始符号可以使用的列表，以及下一个可以使用的开始符号的索引

        self.print_patterns("未进行处理的规则列表")  # 输出当前的规则列表

        self.__remove_epsilon()  # 对规则列表进行去除空产生式
        self.print_patterns("去除空产生式之后的规则列表")  # 输出当前的规则列表

        self.__remove_single()  # 去除规则列表中的单产生式
        self.print_patterns("去除单一产生式之后的规则列表")

        self.__remove__useless_char()
        self.print_patterns("去除了无用符号之后的规则列表")

        self.__convert_to_G1()  # 将规则列表转换为 G1 形式，这是转化为 Chomsky 的中间形式
        self.print_patterns("转化为 Chomsky 范式的中间形态 G1 的规则列表")

        self.__convert_to_GC()  # 将规则列表抓换为 GC 形式，即 Chomsky 形式
        self.print_patterns("转化为 Chomsky 范式后的规则列表")

        self.__for_orders()  # 对规则列表按照顺序进行拓展处理
        self.print_patterns("按顺序排序后的规则列表")

        self.__remove_left_recursion()  # 对规则列表进行消除左递归处理
        self.print_patterns("消除左递归后的规则列表")

        self.__for_lower()  # 对规则列表转换成开始符打头
        self.print_patterns("将规则列表转换为开始符打头后的规则列表")

        if self.__if__equal("S1", "S"):
            self.patterns.pop("S1")
        self.print_patterns("转换之后的格雷巴赫范式")
        # print("Good")  # 一般用来打断点的地方

    def __if__equal(self, first, second):
        if len(self.patterns[first].derive) != len(self.patterns[second].derive):
            return False
        else:
            flag = True
            for i in range(len(self.patterns[first].derive)):
                if flag:
                    if self.patterns[first].derive[i] != self.patterns[second].derive[i]:
                        flag = False
            return flag

    def __remove__useless_char(self):
        """
        去除文法中的无用符号，可以分为两个步骤：1.去除没有引用的开始符 2.去除不能产生终止符号的开始符号
        首先记录了在原文法中的开始符号列表，然后剔除无用符号之后，回复原本的顺序即可
        """
        key_order_list = list(self.patterns.keys())  # 记录原本的开始符号顺序
        self.__remove_no_ref_char()  # 去除没有引用的开始符号
        self.__remove_no_end_char()  # 去除不产生终止符号的开始符号
        temp_dic = {}  # 恢复顺序
        for key in key_order_list:
            if key in self.patterns.keys():
                temp_dic[key] = self.patterns.pop(key)
        self.patterns = temp_dic

    def __remove_no_ref_char(self):
        """
        去除文法中没有被引用的开始符
        """
        char_lists = list(self.patterns.keys())  # type: List[str]
        if "S1" in char_lists:
            char_lists.remove("S1")
        char_lists.remove("S")
        for key in self.patterns.keys():
            for item in self.patterns[key].derive:
                for _ in item:
                    if 'A' <= _ <= 'Z':
                        if _ in char_lists:
                            char_lists.remove(_)
        for no_ref_char in char_lists:
            self.patterns.pop(no_ref_char)

    def __remove_no_end_char(self):
        """
        去除不产生终止符号的开始符
        """
        candidate_dict = {}
        candidate_key = []
        for key in self.patterns.keys():  # 首先处理符号列表，将不能直接产生终结符号的文法加入到 candidate_dict 之中
            if key in ["S", "S1"]:
                continue
            flag = False
            for item in self.patterns[key].derive:
                if item == "*" or item.islower() or item.isdigit():
                    flag = True
                    break
            if not flag:
                candidate_key.append(key)
        for x in candidate_key:
            candidate_dict[x] = self.patterns.pop(x)
        # 检查位于 candidate_dict 中的文法，判断它是否可以间接推导出终结符，即推出还存在文法字典中的开始符号
        self.__check_if_no_end(candidate_dict)

    def __check_if_no_end(self, candidate_dict):
        """
        判断位于 candidate_dict 字典中的文法是否可以间接推出还存在于文法字典中的文法
        即判断其最终是否可以推导出终结符，如果不行，则判定其为无用符号
        :param candidate_dict: 候选字典
        """
        useful_key = []
        for key in candidate_dict.keys():
            for item in candidate_dict[key].derive:
                flag = True
                for _ in item:
                    if "A" <= _ <= 'Z':  # 如果 item 中存在开始符号
                        if _ == key or _ not in self.patterns.keys():  # 若该开始符号不存在于文法字典中或等于自身形成循环推导
                            flag = False  # 则判断该 item 无法帮助候选文法达到终结
                            break
                if flag:  # 遍历 item 中的符号，如果满足达到终结条件，在此处添加至有用符号列表，终止遍历该候选文法
                    useful_key.append(key)
                    break
        for key in useful_key:  # 将候选文法重新加入文法字典中
            self.patterns[key] = candidate_dict.pop(key)
        for key in self.patterns.keys():
            temp_list = self.patterns[key].derive[:]
            for temp_item in self.patterns[key].derive:
                for x in candidate_dict:
                    if x in temp_item:
                        temp_list.remove(temp_item)
                        break
            self.patterns[key].derive = temp_list

    def __get_orders(self) -> Tuple[List[str], int]:
        """
        获取后续所需的列表排序，以及下一个可以使用的作为开始符的索引

        :return: 列表排序，前部分是现有规则的开始符号，后一部分是还没有使用过的符号
        """
        temp = []
        temp.extend(self.patterns.keys())
        idx = len(temp)
        for i in range(26):
            e = chr(ord('A') + i)
            if e not in temp:
                temp.append(e)
        return temp, idx

    def __if_derive_epsilon(self, characters) -> bool:
        """
        判断某个开始字符领导的推导结果中是否包含空即 epsilon
        输入的 characters 表示的是某个规则的推导结果中的 item 的切片是否包含会导致推出空的开始符\n
        :param characters:  推导结果中 item 的一部分切片
        :return:  是否会产生空表示的布尔值
        """
        flag = False
        for _ in characters:
            if _ > 'Z' or _ < 'A':  # 如果存在某个符号并不是开始符，那么这个切片不能被认为可以推出空
                return False
            elif '*' not in self.patterns[_].derive:  # 该开始符领导的规则的推导结果并不包含空
                return False
            elif '*' in self.patterns[_].derive:  # 该开始符领导的规则的推导结果包含空
                flag = True
        return flag

    def __remove_epsilon(self):
        """
        从规则中删除 epsilon 产生式
        遍历所有的规则，首先处理所有能够间接推导出 epsilon 的item，处理方式是：
        遍历 item 的所有切片，如果该切片仅包含开始符号，并且所有符号都能够推出空，则新增一个 item1， 它在 item 中去除了切片
        接着处理所有能够直接推出 epsilon 的规则，处理方式是直接从推导结果列表中删除 epsilon
        """
        for x in self.patterns.keys():  # 遍历规则
            added_list = []  # 需要增加到推导结果中的列表
            for item in self.patterns[x].derive:  # 遍历推导结果中的每一个 item
                for left in range(len(item)):  # 使用双指针，遍历所有的切片
                    for right in range(left, len(item)):
                        if self.__if_derive_epsilon(item[left:right + 1]):  # 使用函数，判断是否会产生空产生式
                            sub_str = item[:left] + item[right + 1:]  # 从 item 中除去切片
                            if len(sub_str) != 0:  # 防止切片为空
                                added_list.append(sub_str)
            self.patterns[x].derive.extend(added_list)  # 将 added_list 列表中的结果合并到推导结果中
        for x in self.patterns.keys():  # 直接删除 epsilon
            if '*' in self.patterns[x].derive:
                self.patterns[x].derive.remove('*')
            self.patterns[x].derive = sorted(list(set(self.patterns[x].derive)))  # 简单粗暴的去重

    def __remove_single(self):
        """
        从规则列表中删除单一产生式，简而言之便是将推导结果中所有的开始符能够推出的所有结果归并到该规则中
        """
        for x in self.patterns.keys():
            added_list = []  # 最终需要归并到当前推导式中的结果
            removed_list = []
            for item in self.patterns[x].derive:
                if len(item) == 1 and 'A' <= item <= 'Z':  # 要求该 item 必须是单个开始符
                    added_list.extend(self.__get_all_not_single_derive(item, x))  # 将该开始符能推导出来的结果归并到 added_list中
                    removed_list.append(item)
            for item in removed_list:
                self.patterns[x].derive.remove(item)
            self.patterns[x].derive.extend(added_list)  # 归并
            self.patterns[x].derive = sorted(list(set(self.patterns[x].derive)))  # 粗暴去重并且排序

    def __get_all_not_single_derive(self, key, parent_key) -> List[str]:
        """
        递归获取所有可能的推导结果

        :param key: 从 key 开始符开始深度遍历，罗列所有的可能推导结果
        :return:    返回所有可能的推导结果
        """
        result = []
        for x in self.patterns[key].derive:
            if len(x) == 1 and 'A' <= x <= 'Z' and x != parent_key:  # 如果是开始符，那么就深度递归
                result.extend(self.__get_all_not_single_derive(x, parent_key))  # 进行递归
            elif x != parent_key:
                result.append(x)  # 如果不是开始符，那么就加入列表中
        return list(set(result))  # 粗暴的去重

    def __convert_to_G1(self):
        """
        将规则列表转换成 G1 形式，具体的处理方法是：

        查看所有的 item 中长度大于等于 2 的情形，将小写字母用大写字母表示\n
        同时新增转换关系，构造新规则，表示从该 大写字母 映射到 小写字母 的情形

        """
        # 因为如果加入到 patterns 中，会在后续被遍历
        added_patterns = []  # 需要新增的规则，暂时还不能添加到 patterns 中
        for x in self.patterns.keys():
            replaced_derive = []  # 该规则可以推导得到的结果，该列表是经过转化 G1 后的结果
            for item in self.patterns[x].derive:
                temp = ""  # 处理每一个 item， 从空开始构造处理之后的 item
                if len(item) >= 2:  # 如果这个 item 的大小大于等于 2 才进行处理
                    for idx in range(len(item)):  # 遍历这个 item
                        if 'a' <= item[idx] <= 'z':  # 如果这个 e 属于 V 集合，那么对其进行处理
                            find = self.__search_in_temp(added_patterns, item[idx])  # 查看是否已经存在新建规则指向该个 e
                            if find != self.empty:  # 若存在新规则
                                temp += find  # 则 item 的这个 e 替换为该规则名字
                            else:
                                temp += self.orders[self.idx]  # 否则新建规则，temp 加入新规则的开始符
                                p = pattern("{}->{}".format(self.orders[self.idx], item[idx]))  # 新建规则
                                added_patterns.append(p)  # 将规则加入到 added_patterns
                                self.idx += 1  # 递增可用开始符的指针
                        else:
                            temp += item[idx]  # 若是开始符号，则不进行处理直接加入
                else:
                    temp = item  # 如果 item 的长度仅为 1 ，则直接将 temp 设为 item
                replaced_derive.append(temp)  # 在替换列表中加入 temp
            self.patterns[x].derive = sorted(replaced_derive)  # 将推导的结果转换为 replaced_derive
        self.__add_patterns(added_patterns)  # 调用函数，将新建的规则加入到类内规则列表中

    def __convert_to_GC(self):
        """
        将规则列表转换成 GC 形式，具体的处理方法有些复杂：

        对于所有的长度大于 2 的 item 进行处理，例如：
            S -> ABC 转化为：
            S -> AD, D -> BC
        逐个递归分解，但是要注意如果有一个 item在分解的时候，之前的其他 item 在分解时已经新建了一个指向 BC 的规则
        那么当前 item 在分解时，不应当再新建规则，应该使用已经创建的规则

        """
        added_patterns = []  # 同上，新建但是暂未加入的规则
        for x in self.patterns.keys():
            replaced_derive = []  # 替换当前规则的派生结果
            for item in self.patterns[x].derive:
                if len(item) <= 2:  # 如果 item 的长度小于等于 2 ，则不作处理
                    replaced_derive.append(item)
                else:
                    is_first = True  # 判断是否是开头，如果是开头，除了新建规则，还需要处理替换
                    rest = item
                    while True:
                        in_dict = False  # 判断剩下的部分是否已经存在于规则或者新建规则中，如果存在，则仅需要简单处理就可以终止循环
                        search_result = self.__search_in_patterns_and_temp(added_patterns, rest[1:])  # 已经创建的规则开始符
                        if search_result != self.empty:
                            if is_first:
                                replaced_derive.append(item[0] + search_result)  # 修改派生
                                break  # 对于这种情况，后续不用继续分解，直接指向了已经创建的规则

                        else:  # 没有对应的新建规则
                            if is_first:
                                start = self.orders[self.idx]  # 获取一个可用的开始符号
                                replaced_derive.append(item[0] + start)  # 是开始部分，故而要修改派生
                                if len(rest[1:]) == 2:
                                    p = pattern("{}->{}".format(start, rest[1:]))  # 如果剩下只剩下两个部分，则规则直接指向剩余部分
                                else:  # 否则还需要递归分解
                                    p = pattern("{}->{}".format(start, rest[1] + self.orders[self.idx + 1]))
                                p.solved_problem = rest[1:]
                                added_patterns.append(p)  # 将规则添加到待添加列表中
                                self.idx += 1
                                is_first = False
                            else:
                                start = self.orders[self.idx]
                                self.idx += 1
                                if len(rest[1:]) == 2:  # 同上，如若只剩下最后两个字符，则直接指向
                                    p = pattern("{}->{}".format(start, rest[1:]))
                                else:
                                    # 判断是否已经有新建的规则可以使用
                                    search_result = self.__search_in_patterns_and_temp(added_patterns, rest[2:])
                                    if search_result != self.empty:  # 若有，则直接使用，停止循环
                                        p = pattern("{}->{}".format(start, search_result))
                                        in_dict = True
                                    else:  # 若无，则新建规则，继续循环
                                        p = pattern("{}->{}".format(start, self.orders[self.idx]))
                                p.solved_problem = rest[1:]
                                added_patterns.append(p)
                        if in_dict:
                            break
                        rest = rest[1:]
                        if len(rest) == 2:  # 只剩下 2 的长度，则无需要再继续循环，结束
                            break
            self.patterns[x].derive = sorted(replaced_derive)  # 更改派生
        self.__add_patterns(added_patterns)  # 将待添加规则添加到类内自身规则列表中

    def __add_s1_pattern(self):
        """
        增加 S1 规则，并且进行排序，让 S1 处于首位
        """
        p = pattern()
        p.name = "S1"
        p.derive = self.patterns["S"].derive[:]  # 做一个深拷贝，否则指针会指向同一个列表
        new_dict = {"S1": p}
        for x in self.patterns.keys():
            new_dict[x] = self.patterns[x]
        self.patterns = new_dict

    def __remove_left_recursion(self):
        """
        对规则列表进行消除左递归处理，其具体处理逻辑如下所示：
        遍历所有的规则，检查派生列表，查看是否存在开头和该规则开始符号相同的 item：
            若有，则停止检查当前列表，获取一个可用的新开始符号，进行处理。从当前规则的派生列表中去除符合条件的 item ，保存列表中剩余的 item ，
            并且制剩余的 item ，在其末尾添加新开始符号。第二步，新建规则，以新开始符为开始符，添加上个规则中被去除的 item 的 [1:]部分，并且
            拷贝一份，在其末尾添加新的开始符号
            若无，则不对当前规则进行处理
        以下给出示例：
            A -> Aα | B 经过去除左递归得到
            A -> BZ | B
            Z -> α | αZ
        """
        added_patterns = []  # 保存当前不能放入 self.patterns 的新建规则
        for x in self.patterns.keys():  # 遍历规则
            flag = False  # 指示是否存在开头和开头符号相同的 item
            for item in self.patterns[x].derive:
                if item[0] == x:
                    flag = True  # 若存在，则 flag 置为 True 并且中断
                    break
            if flag:  # 若存在如上描述的 item，则进行处理
                new_derive = []  # 新规则的派生列表
                replaced_derive = []  # 取代当前规则的派生列表
                start = self.orders[self.idx]  # 提前获取一个可用的开始符号
                self.idx += 1
                for item in self.patterns[x].derive:  # 按照函数描述中的方式进行处理
                    if item[0] == x and len(item) > 1:
                        if len(item[1:]) != 0:
                            new_derive.append(item[1:])
                        new_derive.append(item[1:] + start)
                    else:
                        replaced_derive.append(item)
                        replaced_derive.append(item + start)
                self.patterns[x].derive = replaced_derive  # 替换当前规则的派生列表
                p = pattern()  # 新建规则实例
                p.name = start
                p.derive = new_derive
                added_patterns.append(p)
        self.__add_patterns(added_patterns)  # 将产生的所有新建规则加入到类内规则列表中

    def __for_orders(self):
        """
        按照顺序进行扩展，其处理方法是：
        按照规则开始符号的优先顺序，查找派生结果中每个 item 的第一个字符是否优先于该规则的开始符
        如果有，则将该字符展开，就是转换为它的派生，递归，直至最后的结果中每个 item 的第一个字符都低于该规则的开始符号
        不处理 S1 和 S
        """
        length = len(self.patterns.keys())  # 规则字典的键的列表
        for i in range(length):
            key = list(self.patterns.keys())[length - 1 - i]  # 从后向前，从优先级低的开始遍历
            if key == "S" or key == "S1":  # 不处理 S 和 S1 开头的规则
                continue
            p = self.patterns[key]  # 获取该规则类
            replaced_derive = []  # 准备用于替换的列表
            for item in p.derive:
                if item[0] < key or item[0] == 'S':  # 检查是否存在优先级比 key 高的
                    result = self.__process_orders(item, key)  # 若有，则调用 __process_orders 方法，它返回替换的字符列表
                    replaced_derive.extend(result)  # 融合
                else:
                    replaced_derive.append(item)  # 如果优先级比 key 低，那么直接加入
            p.derive = replaced_derive  # 替换列表

    def __process_orders(self, e: str, key: str) -> List[str]:
        """
        该方法可用于递归处理规则中有限符高的派生

        :param e: 需要处理的派生
        :param key: 该规则的键 key
        :return: 返回处理结果的列表
        """
        result = []
        for item in self.patterns[e[0]].derive:
            if 'a' <= item[0] <= 'z' or (item[0] >= key and item[0] != 'S'):  # 如果属于字符表或者优先级更低的派生，则直接拼接后续字符，加入result
                result.append(item + e[1:])
            else:
                recursion_result = self.__process_orders(item, key)  # 如果仍然优先级比 key 更高，则递归处理
                processed_recursion_result = [x + e[1:] for x in recursion_result]  # 对处理结果进行拼接，并且全部添加到 result中
                result.extend(processed_recursion_result)
        return result

    def __for_lower(self):
        """
        将规则转换为小写字母打头
        和处理顺序的相似
        """
        length = len(self.patterns.keys())
        for i in range(length):
            key = list(self.patterns.keys())[length - 1 - i]
            p = self.patterns[key]
            replaced_derive = []
            for item in p.derive:
                if 'a' <= item[0] <= 'z':
                    replaced_derive.append(item)
                else:
                    result = self.__process_lower(item)
                    replaced_derive.extend(result)
            p.derive = replaced_derive

    def __process_lower(self, e: str) -> List[str]:
        """
        和上面递归处理顺序的方法类似
        :param e: 需要处理的派生
        :return: 返回处理之后的结果
        """
        result = []
        for item in self.patterns[e[0]].derive:
            if 'a' <= item[0] <= 'z':
                result.append(item + e[1:])
            else:
                recursion_result = self.__process_lower(item)
                processed_recursion_result = [x + e[1:] for x in recursion_result]
                result.extend(processed_recursion_result)
        return result

    def __search_in_patterns_and_temp(self, temps: List[pattern], target: str) -> str:
        """
        在类内自身规则列表和临时列表中同时查找
        是否会有规则派生出 target
        如若有，则返回该规则的开始符
        如没有，则返回 self.empty

        :param temps: 临时列表
        :param target: 需要查找的派生结果
        :return: self.empty 或者是规则的开始名称
        """
        for x in self.patterns.keys():  # 在类内规则列表中进行查找
            if target in self.patterns[x].derive and len(self.patterns[x].derive) == 1:
                return x
        # for x in temps:  # 在临时列表中进行查找
        #     if target in x.derive:
        #         return x.name
        for x in temps:
            if target == x.solved_problem:
                return x.name
        return self.empty

    def __search_in_temp(self, temps: List[pattern], target: str) -> str:
        """
        在临时列表中进行查找

        :param temps: 临时列表
        :param target: 需要查找的派生结果
        :return: self.empty 或者是规则的开始名称
        """
        for x in temps:  # 在临时列表中进行查找
            if target in x.derive:
                return x.name
        # for x in temps:
        #     if target == x.solved_problem:
        #         return x.name
        return self.empty

    def __add_patterns(self, patterns: List[pattern]):
        """
        将临时列表中的规则添加到类内规则列表中

        :param patterns: 临时规则列表
        """
        for x in patterns:
            self.patterns[x.name] = x

    def print_patterns(self, info):
        """
        打印 info 之后，输出类内规则列表
        :param info:  需要打印的提示信息
        """
        print(info)
        for x in self.patterns.keys():
            print("%2s" % x, " -> ", self.patterns[x].derive)
        print("\n")


class pda:
    """
    下推自动机
    """

    def __init__(self, normal_greibach: greibach, text: str):
        if "S1" in normal_greibach.patterns.keys():
            self.stack = ["S1"]  # 堆栈，用来保存下推结果
        else:
            self.stack = ["S"]
        self.greibach = normal_greibach  # type: greibach
        self.sentence = text  # 需要处理的语句
        self.idx = 0  # 当前处理到语句的索引位置
        self.flag = False  # 是否已经判断完毕

    def analyze(self):
        """
        暴露在外的用于处理的函数，调用 __analyze() 方法
        通过判断 self.flag 来检查是否属于该文法
        """
        self.__analyze()
        if self.flag:
            print("该句子属于此文法！")
        else:
            print("该句子不属于此文法！")

    def __analyze(self):
        """
        下推自动机用于分析语句是否符合当前语法
        其实就类似于深度优先的搜索
        """
        if not self.flag:  # 判断当前是否已经完成判断
            if len(self.stack) == 0 and self.idx != len(self.sentence):  # 如果栈空了且还没有遍历完语句的话，说明该方向错误
                return
            top = self.stack.pop()  # 弹出栈顶元素
            length = len(self.greibach.patterns[top].derive)
            for i in range(length):  # 遍历该开始符的派生列表
                item = self.greibach.patterns[top].derive[i]  # 某个派生结果 item
                if item[0] == self.sentence[self.idx]:  # 需要注意第一个字符得和语句当前检查位置的字符相同
                    temp = list(item[1:])  # 将派生结果 item 剩余部分逆序放入栈中
                    self.idx += 1  # 索引位置向后递增
                    temp.reverse()
                    temp_stack = self.stack[:]  # 保存加入前的栈情况，用于在完成后续的递归之后复原
                    self.stack.extend(temp)
                    state = self.__check()  # 判断当前结果是否满足结束条件，返回状态位
                    if state == 0:  # 0 表示，已经完成判断
                        self.flag = True
                    elif state == 1:  # 1 表示 idx 已经到最后，但是栈中不为空，退栈，idx 递减
                        self.idx -= 1
                        # self.stack = temp_stack
                        # continue
                    else:
                        self.__analyze()  # 2 表示还应该继续向下递推
                        self.idx -= 1
                    self.stack = temp_stack  # 将状态回复

    def __check(self) -> int:
        """
        检查是否满足结束的条件
        """
        if self.idx == len(self.sentence):
            if len(self.stack) == 0:
                return 0
            elif len(self.stack) > 0:
                return 1
        else:
            return 2


if __name__ == "__main__":
    g = greibach()  # 构造格雷巴赫范式
    sentence = input("请输入想要分析的语句\n")
    pda_machine = pda(g, sentence)  # 构造下推自动机
    pda_machine.analyze()  # 下推自动机开始分析，语句是否满足文法

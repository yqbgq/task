"""
形式语言与自动机课程作业：
使用图灵机完成幂运算
@Author 黄伟
"""

from typing import Dict

# 图灵机规则存放位置
patterns_path = "patterns.txt"


def build_patterns() -> Dict[str, Dict]:
    """
    从转移函数文件中读取所有的转移函数
    :return: 转移函数的字典
    """
    patterns = {}  # type: Dict[str, Dict]
    with open(patterns_path) as f:
        line = f.readline()
        while line:
            elements = line[:-1].split(" ")  # 去除每行末尾的换行符，并且使用空格进行分割
            item = elements[0]  # 状态名称
            if item not in patterns.keys():
                patterns[item] = {}
            patterns[item][elements[1]] = elements[2:]
            line = f.readline()
    return patterns


class turing:
    def __init__(self, x, y):
        self.x = int(x)     # 底数 X
        self.y = int(y)     # 指数 Y
        self.tape = self.__build_tape()
        self.patterns = build_patterns()
        self.idx = 1
        self.condition = "q0"

    def run(self):
        """
        让图灵机它跑起来！！！
        """
        while self.condition != "qf":  # 如若当前的状态不为 qf 则表示还需要进行处理，否则停机
            tape_e = self.tape[self.idx]  # 获取当前指针指向的元素

            # 获取对应的处理方法元组，包含内容：转移之后的机器状态，当前元素改变为何种元素以及指针移动方向
            solution_tuple = self.patterns[self.condition][tape_e]

            # # 打印纸带，包含 debug 信息
            # self.print_tape(debug=True)
            # 打印纸带，不包含其他信息
            self.print_tape(debug=False)

            self.condition = solution_tuple[0]      # 自动机的内部状态

            # 更新纸带信息，进行元素的替换
            self.tape = self.tape[:self.idx] + solution_tuple[1] + self.tape[self.idx + 1:]

            # 移动指针的方向
            if solution_tuple[2] == "r":
                self.idx += 1
            else:
                self.idx -= 1

    def __build_tape(self) -> str:
        """
        构造最初的带
        :return: 构造完成的最初带
        """
        tape = "*"
        for _ in range(self.x):
            tape += "1"
        tape += "*"
        for _ in range(self.y):
            tape += "1"
        for _ in range(pow(self.x, self.y) + 10):
            tape += "*"
        return tape

    def print_tape(self, debug=False):
        """
        打印纸带，根据 debug 参数，设置是否为 debug 模式
        """
        if debug:
            tape_e = self.tape[self.idx]
            solution_tuple = self.patterns[self.condition][tape_e]
            print(self.tape, "对应处理", self.condition, tape_e, solution_tuple)
        else:
            print(self.tape)

    def get_result(self):
        res = self.tape.split("*")
        return res[1], res[2], res[3]


if __name__ == "__main__":
    num_x = input("请输入 X 的值 \n")
    num_y = input("请输入 Y 的值 \n")
    # num_x = 5
    # num_y = 3
    turing_machine = turing(num_x, num_y)
    turing_machine.run()
    turing_machine.print_tape()
    result_row = turing_machine.get_result()
    result = [len(x) for x in result_row]

    print()
    print("x: ", result_row[0], " y: ", result_row[1], " z: ", result_row[2])
    print("x: ", result[0], " y: ", result[1], " z: ", result[2])


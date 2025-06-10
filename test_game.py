import unittest
import numpy as np
import numpy.testing as npt

#from game import Game


class TestGameRules(unittest.TestCase):

    def setUp(self):
        self.game = Game()
        self.B = self.game.PLAYER_BLACK
        self.W = self.game.PLAYER_WHITE
        self.E = self.game.EMPTY_SLOT

    def test_simple_horizontal_line(self):
        """测试场景1：一次简单的横向连线"""
        print("\nRunning test: simple_horizontal_line")

        # 1. Arrange: 假设黑方在 (0,0) 和 (0,1) 已有棋子
        self.game.current_player = self.B
        self.game.board_pieces[0, 0] = self.B
        self.game.board_pieces[0, 1] = self.B

        # 2. Act: 在 (0,2) 落子，形成连线
        self.game.board_pieces[0, 2] = self.B
        lines_made = self.game._check_and_process_lines(r=0, c=2)

        # 3. Assert
        self.assertTrue(lines_made, "方法应返回True，因为形成了连线")

        expected_pieces = np.zeros_like(self.game.board_pieces)
        npt.assert_array_equal(self.game.board_pieces, expected_pieces, "连线棋子应被移除")

        expected_territory = np.zeros_like(self.game.board_territory)
        expected_territory[0, :] = self.B
        npt.assert_array_equal(self.game.board_territory, expected_territory, "整条线都应成为领地")

    def test_no_line_formed(self):
        """【已修正】测试场景2：落子后没有形成连线"""
        print("\nRunning test: no_line_formed (Corrected)")

        # 1. Arrange: 棋盘上散布一些棋子，但落子后无法形成连线
        self.game.current_player = self.B
        self.game.board_pieces[0, 0] = self.B
        self.game.board_pieces[5, 5] = self.W

        # 记录落子前的状态
        initial_pieces = self.game.board_pieces.copy()

        # 2. Act: 在一个无关位置 (2,2) 落子
        self.game.board_pieces[2, 2] = self.B
        lines_made = self.game._check_and_process_lines(r=2, c=2)

        # 3. Assert
        self.assertFalse(lines_made, "方法应返回False，因为没有形成连线")

        # 验证领地没有变化
        npt.assert_array_equal(self.game.board_territory, np.zeros_like(self.game.board_territory),
                               "没有连线，领地不应改变")

        # 验证棋盘状态：应该只是比初始状态多了(2,2)这一颗棋子
        expected_pieces = initial_pieces
        expected_pieces[2, 2] = self.B
        npt.assert_array_equal(self.game.board_pieces, expected_pieces, "棋盘上应该只增加了刚才落下的那一子")

    def test_territory_expansion_blocked(self):
        """测试场景4：领地扩展被对方棋子阻挡"""
        print("\nRunning test: territory_expansion_blocked")

        self.game.current_player = self.B
        self.game.board_pieces[2, 2] = self.B
        self.game.board_pieces[2, 3] = self.B
        self.game.board_pieces[2, 7] = self.W  # 对方的阻挡棋子

        self.game.board_pieces[2, 4] = self.B  # 模拟落子
        lines_made = self.game._check_and_process_lines(r=2, c=4)

        self.assertTrue(lines_made)

        expected_territory = np.zeros_like(self.game.board_territory)
        expected_territory[2, 0:7] = self.B
        npt.assert_array_equal(self.game.board_territory, expected_territory, "领地扩展应在对方棋子前停止")

        expected_pieces = np.zeros_like(self.game.board_pieces)
        expected_pieces[2, 7] = self.W
        npt.assert_array_equal(self.game.board_pieces, expected_pieces, "只有对方的棋子应该保留")

    def test_double_line_formation(self):
        """测试场景3：一次落子同时形成两条线 (十字)"""
        print("\nRunning test: double_line_formation")

        self.game.current_player = self.B
        self.game.board_pieces[0, 2] = self.B
        self.game.board_pieces[1, 1] = self.B
        self.game.board_pieces[1, 3] = self.B
        self.game.board_pieces[2, 2] = self.B

        self.game.board_pieces[1, 2] = self.B  # 模拟在十字中心落子
        lines_made = self.game._check_and_process_lines(r=1, c=2)

        self.assertTrue(lines_made)

        expected_pieces = np.zeros_like(self.game.board_pieces)
        npt.assert_array_equal(self.game.board_pieces, expected_pieces, "所有参与连线的棋子都应被移除")

        expected_territory = np.zeros_like(self.game.board_territory)
        expected_territory[:, 2] = self.B
        expected_territory[1, :] = self.B
        npt.assert_array_equal(self.game.board_territory, expected_territory, "两条线的领地都应被占据")


if __name__ == '__main__':
    unittest.main()
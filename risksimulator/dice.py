import random
from typing import Tuple, List


class Dice:
    """Dice with a maximum and a minimum value."""

    def __init__(self, minimal_value: int = 1, maximal_value: int = 6):
        self.minimal_value = minimal_value
        self.maximal_value = maximal_value

    def throw_dice(self) -> int:
        """
        Throw the dice.

        :return:  A random result in between the defined maximum and minimum.
        :rtype: int
        """

        return random.randint(self.minimal_value, self.maximal_value)


class Attacker:
    """The risk attacker dice throw."""

    def __init__(self, minimal_value, maximal_value):
        self.minimal_value = minimal_value
        self.maximal_value = maximal_value

        self.first_dice, self.second_dice, self.third_dice = [
            Dice(minimal_value=self.minimal_value, maximal_value=self.maximal_value)
            for _ in range(3)
        ]

    def throw_attack(self):
        attacker_result = sorted(
            [
                self.first_dice.throw_dice(),
                self.second_dice.throw_dice(),
                self.third_dice.throw_dice(),
            ],
            reverse=True,
        )
        top_two_results = attacker_result[:2]

        return top_two_results


class Defender:
    """The risk defender dice throw."""

    def __init__(self, minimal_value, maximal_value):
        self.minimal_value = minimal_value
        self.maximal_value = maximal_value

        self.first_dice, self.second_dice = [
            Dice(self.minimal_value, self.maximal_value) for _ in range(2)
        ]

    def throw_defense(self):
        defender_result = sorted(
            [self.first_dice.throw_dice(), self.second_dice.throw_dice()], reverse=True
        )
        return defender_result


class MonteCarloSimulator:
    def __init__(
        self, defender_range: Tuple[int, int], attacker_range: Tuple[int, int]
    ):
        self.defender = Defender(
            minimal_value=defender_range[0], maximal_value=defender_range[1]
        )
        self.attacker = Attacker(
            minimal_value=attacker_range[0], maximal_value=attacker_range[1]
        )

    def _throw_attack_and_defense(self):
        defender_result = self.defender.throw_defense()
        attacker_result = self.attacker.throw_attack()
        return attacker_result, defender_result

    @staticmethod
    def _compare_dice_and_return_count(attacker_result, defender_result):
        attacker_count, defender_count = 0, 0
        if defender_result >= attacker_result:
            attacker_count = attacker_count - 1
        else:
            defender_count = defender_count - 1
        return attacker_count, defender_count

    def _compare_entire_throw_and_keep_count(
        self, attacker_result, defender_result, attacker_count, defender_count
    ):
        # First dice.
        (
            attacker_first_score,
            defender_first_score,
        ) = self._compare_dice_and_return_count(
            attacker_result=attacker_result[0], defender_result=defender_result[0]
        )
        (
            attacker_second_score,
            defender_second_score,
        ) = self._compare_dice_and_return_count(
            attacker_result=attacker_result[1], defender_result=defender_result[1]
        )

        # Second dice.
        attacker_count = attacker_count + attacker_first_score + attacker_second_score
        defender_count = defender_count + defender_first_score + defender_second_score

        return attacker_count, defender_count

    def run_bootstrap(self, nruns=10000):
        # Set initial count and run number.
        count_attacker, count_defender, run_number = 0, 0, 0

        while run_number < nruns:
            attacker_result, defender_result = self._throw_attack_and_defense()
            (
                interim_attacker,
                interim_defender,
            ) = self._compare_entire_throw_and_keep_count(
                attacker_result=attacker_result,
                defender_result=defender_result,
                attacker_count=count_defender,
                defender_count=count_defender,
            )
            count_attacker, count_defender = (
                count_attacker + interim_attacker,
                count_defender + interim_defender,
            )
            run_number = run_number + 1
        return count_attacker, count_defender

    #     if run_number % 100000 == 0:
    #         print(run_number)
    #         print("Attacker-Defender:", count_attacker, "-", count_defender)
    #
    #         count = count_attacker + count_defender
    #         per_attacker, per_defender = count_attacker / count, count_defender / count
    #         print(
    #             "{:0.2f}% loss for the attacker and {}% loss for the defender".format(
    #                 per_attacker, per_defender
    #             )
    #         )
    #
    # else:
    #     print("Alia iacta est!")
    #     print("Attacker-Defender:", count_attacker, "-", count_defender)


if __name__ == "__main__":
    simulator = MonteCarloSimulator(defender_range=(0, 6), attacker_range=(0, 6))
    c_at, c_de = simulator.run_bootstrap(nruns = 4)
    print(c_at, c_de)
    print("Alia iacta est!")

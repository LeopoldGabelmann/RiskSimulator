import logging
import random
import statistics
from typing import Tuple

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _absolute_loss_to_percentage(count_attacker, count_defender):
        count = count_attacker + count_defender
        per_attacker, per_defender = (
            count_attacker / count * 100,
            count_defender / count * 100,
        )
        return per_attacker, per_defender

    def _compare_entire_throw_and_keep_count(self, attacker_result, defender_result):
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
        attacker_count = attacker_first_score + attacker_second_score
        defender_count = defender_first_score + defender_second_score

        return attacker_count, defender_count

    def run_one_bootstrap(self, nruns=10000, verbose=0):
        # Set initial count and run number.
        count_attacker, count_defender, run_number = 0, 0, 1

        while run_number < nruns:
            attacker_result, defender_result = self._throw_attack_and_defense()
            (
                interim_attacker,
                interim_defender,
            ) = self._compare_entire_throw_and_keep_count(
                attacker_result=attacker_result, defender_result=defender_result,
            )
            count_attacker, count_defender = (
                count_attacker + interim_attacker,
                count_defender + interim_defender,
            )

            run_number = run_number + 1

        if run_number == nruns and verbose == 1:
            per_attacker, per_defender = self._absolute_loss_to_percentage(
                count_attacker, count_defender
            )
            logger.info(
                f"The Attacker:Defender count for this bootstrap is "
                f"{count_attacker}:{count_defender}, i.e. {per_attacker:3.2f}% "
                f"loss for the attacker and {per_defender:3.2f}% loss for the "
                f"defender"
            )

        return count_attacker, count_defender

    def run_bootstrap_monte_carlo(self, n_bootstraps=1000, nruns=10000, verbose=0):
        bootstrap_count = 0

        bootstrap_counts = []
        while bootstrap_count < n_bootstraps:
            if bootstrap_count % verbose == 0:
                print_bootstrap = 1
                logger.info(f"Calculating bootstrap number {bootstrap_count}.")
            else:
                print_bootstrap = 0

            bootstrap_counts.append(
                (self.run_one_bootstrap(nruns=nruns, verbose=print_bootstrap))
            )
            bootstrap_count = bootstrap_count + 1

        unzipped = list(zip(*bootstrap_counts))
        mean_attacker = statistics.mean(unzipped[0])
        mean_defender = statistics.mean(unzipped[1])
        per_attacker, per_defender = self._absolute_loss_to_percentage(
            mean_attacker, mean_defender
        )
        logger.info(
            f"Alia iacta est. \n The mean result (Attacker:Defender) over all "
            f"bootstraps is {mean_attacker}:{mean_defender}, i.e. "
            f"{per_attacker:3.2f}% loss for the attacker and "
            f"{per_defender:3.2f}% loss for the defender"
        )

        return mean_attacker, mean_defender


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(levelname)-8s - %(name)s\t- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simulator = MonteCarloSimulator(defender_range=(0, 6), attacker_range=(0, 6))
    c_at, c_de = simulator.run_one_bootstrap(nruns=10, verbose=1)
    simulator.run_bootstrap_monte_carlo(n_bootstraps=1000, nruns=10000, verbose=50)

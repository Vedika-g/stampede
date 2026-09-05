class StampedeRiskEngine:

    def __init__(self):

        self.people_weight = 0.30
        self.density_weight = 0.35
        self.movement_weight = 0.35

        self.high_risk_frames = 0
        self.warning_frames = 0

        self.previous_score = 0.0

    # ========================================================
    # PEOPLE SCORE
    # ========================================================

    def people_score(self, count):

        if count < 20:
            return 0

        elif count < 50:
            return 30

        elif count < 100:
            return 60

        elif count < 150:
            return 80

        return 100

    # ========================================================
    # DENSITY SCORE
    # ========================================================

    def density_score(self, count):

        if count < 20:
            return 0

        elif count < 50:
            return 30

        elif count < 100:
            return 60

        elif count < 150:
            return 80

        return 100

    # ========================================================
    # MOVEMENT SCORE
    # ========================================================

    def movement_score(self, movement):

        if movement < 1.5:
            return 0

        elif movement < 2.5:
            return 30

        elif movement < 4.0:
            return 60

        elif movement < 6.0:
            return 80

        return 100

    # ========================================================
    # RISK CALCULATION
    # ========================================================

    def calculate_risk(
        self,
        people_count,
        csrnet_count,
        movement,
        sudden_movement=False
    ):

        people = self.people_score(
            people_count
        )

        density = self.density_score(
            csrnet_count
        )

        movement_score = self.movement_score(
            movement
        )

        # ----------------------------------------------------
        # BASE RISK
        # ----------------------------------------------------

        risk_score = (

            people * self.people_weight

            +

            density * self.density_weight

            +

            movement_score * self.movement_weight

        )

        # ----------------------------------------------------
        # SUDDEN MOVEMENT BONUS
        # ----------------------------------------------------

        if sudden_movement:

            risk_score += 10

        risk_score = min(
            100,
            round(risk_score, 2)
        )

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if risk_score < 40:

            risk_level = "LOW"

        elif risk_score < 70:

            risk_level = "MEDIUM"

        else:

            risk_level = "HIGH"

        # ----------------------------------------------------
        # HIGH-RISK PERSISTENCE
        # ----------------------------------------------------

        if risk_level == "HIGH":

            self.high_risk_frames += 1

        else:

            self.high_risk_frames = 0

        # ----------------------------------------------------
        # WARNING
        # ----------------------------------------------------

        warning = (
            self.high_risk_frames >= 5
        )

        # ----------------------------------------------------
        # WARNING MESSAGE
        # ----------------------------------------------------

        if warning:

            warning_message = (
                "Potential stampede detected. "
                "Sustained high-risk crowd conditions."
            )

        elif risk_level == "HIGH":

            warning_message = (
                "High crowd risk detected. "
                "Monitoring conditions."
            )

        elif risk_level == "MEDIUM":

            warning_message = (
                "Increasing crowd activity detected."
            )

        else:

            warning_message = (
                "Crowd conditions normal."
            )

        # ----------------------------------------------------
        # PERSISTENCE
        # ----------------------------------------------------

        # Your YOLO runs every 3 frames.
        # Assuming approximately 30 FPS video,
        # each risk calculation represents ~0.1 sec.

        persistence = (
            self.high_risk_frames * 0.1
        )

        self.previous_score = risk_score

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {

            "people_score": people,

            "density_score": density,

            "movement_score": movement_score,

            "risk_score": risk_score,

            "risk_level": risk_level,

            "warning": warning,

            "warning_message": warning_message,

            "persistence": persistence

        }
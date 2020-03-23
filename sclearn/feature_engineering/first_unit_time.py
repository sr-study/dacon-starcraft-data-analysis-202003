import numpy as np

from .extract_game_states import GameState


class FirstUnityTime(GameState):
    Train_Morp_Build = ['(1360) - TrainSCV', '(1021) - BuildSupplyDepot', '(1022) - BuildRefinery', '(1023) - BuildBarracks', '(1020) - BuildCommandCenter', '(13E0) - TrainMarine', '(102A) - BuildFactory', '(1261) - BuildBarracksReactor', '(12A0) - BuildFactoryTechLab', '(1401) - BuildSiegeTank', '(102B) - BuildStarport', '(1024) - BuildEngineeringBay', '(1418) - BuildWidowMine', '(1420) - TrainMedivac', '(1260) - BuildBarracksTechLab', '(15E0) - TrainProbe', '(1541) - BuildPylon', '(1543) - BuildGateway', '(1542) - BuildAssimilator', '(13E1) - TrainReaper', '(154E) - BuildCyberneticsCore', '(1586) - TrainAdept', '(2720) - TrainMothershipCore', '(154D) - BuildRoboticsFacility', '(15C0) - TrainWarpPrism', '(154C) - BuildRoboticsBay', '(12E0) - BuildStarportTechLab', '(1544) - BuildForge', '(15D2) - TrainDisruptor', '(1AC6) - TrainAdept', '(1540) - BuildNexus', '(1546) - BuildTwilightCouncil', '(1025) - BuildMissileTurret', '(15C2) - TrainColossus', '(15C1) - TrainObserver', '(1820) - MorphDrone', '(16E3) - BuildSpawningPool', '(1822) - MorphOverlord', '(1E60) - TrainQueen', '(1821) - MorphZergling', '(1581) - TrainStalker', '(16E0) - BuildHatchery', '(16ED) - BuildRoachWarren', '(15A8) - TrainOracle', '(16E4) - BuildEvolutionChamber', '(16EF) - BuildSporeCrawler', '(2120) - BuildCreepTumor', '(16E2) - BuildExtractor', '(4AE0) - BuildOracleStasisTrap', '(15A4) - TrainVoidRay', '(16E5) - BuildHydraliskDen', '(1545) - BuildFleetBeacon', '(1823) - MorphHydralisk', '(1547) - BuildPhotonCannon', '(1426) - TrainLiberator', '(13E3) - TrainMarauder', '(154A) - BuildTemplarArchive', '(15C3) - TrainImmortal', '(1405) - BuildHellion', '(1421) - TrainBanshee', '(FE0) - CancelBuilding', '(1580) - TrainZealot', '(1549) - BuildStargate', '(1026) - BuildBunker', '(12A1) - BuildFactoryReactor', '(15A0) - TrainPhoenix', '(1829) - MorphRoach', '(16EE) - BuildSpineCrawler', '(102D) - BuildArmory', '(1BA0) - MorphToOverseer', '(102F) - BuildFusionCore', '(154B) - BuildDarkShrine', '(1422) - TrainRaven', '(1424) - TrainViking', '(16E8) - BuildInfestationPit', '(16EA) - BuildBanelingNest', '(920) - TrainBaneling', '(16E6) - BuildSpire', '(103E) - CancelTerranBuilding', '(4020) - MorphToRavager', '(12E1) - BuildStarportReactor', '(1407) - TrainCyclone', '(1824) - MorphMutalisk', '(1585) - TrainSentry', '(15A2) - TrainCarrier', '(5640) - MorphToTransportOverlord', '(2140) - BuildAutoTurret', '(1404) - BuildThor', '(13E2) - TrainGhost', '(1583) - TrainHighTemplar', '(FE1) - HaltBuilding', '(182B) - MorphCorruptor', '(16E9) - BuildNydusNetwork', '(1028) - BuildSensorTower', '(1029) - BuildGhostAcademy', '(182E) - MorphSwarmHost', '(2180) - BuildNydusCanal', '(1406) - BuildBattleHellion', '(1423) - TrainBattlecruiser', '(15A9) - TrainTempest', '(1780) - MorphToGreaterSpire', '(1840) - MorphToBroodLord', '(1BA1) - CancelMorphToOverseer', '(1480) - TrainNuke', '(182A) - MorphInfestor', '(1584) - TrainDarkTemplar', '(4040) - MorphToLurker', '(16E7) - BuildUltraliskCavern', '(1826) - MorphUltralisk', '(9A0) - BuildPointDefenseDrone', '(1660) - TrainInterceptor']
    unity_set = set(Train_Morp_Build)

    def init(self):
        self.first_time = {
            0: {unit: 99999 for unit in FirstUnityTime.unity_set},
            1: {unit: 99999 for unit in FirstUnityTime.unity_set},
        }

    def update(self, game_id, time, player, species, event, event_contents):
        if event == 'Ability':
            contents = FirstUnityTime.parse_contents(event_contents)
            ability = contents[0]
            if ability in FirstUnityTime.unity_set:
                time= FirstUnityTime.min_to_sec(time)
                if  time < self.first_time[player][ability]:
                        self.first_time[player][ability] = time

    def to_dict(self):
        p0_counts = {f'p0_first_time_{ability}': self.first_time[0][ability] for ability in FirstUnityTime.unity_set}
        p1_counts = {f'p1_first_time_{ability}': self.first_time[1][ability] for ability in FirstUnityTime.unity_set}
        delta_counts = {f'delta_first_time_{ability}': self.first_time[0][ability] - self.first_time[1][ability] for ability in FirstUnityTime.unity_set}
        return {**p0_counts, **p1_counts, **delta_counts}

    @staticmethod
    def parse_contents(event_contents):
        contents = event_contents.split(';')
        return contents
    
    @staticmethod
    def min_to_sec(t):
        m = int(t)
        s = (t - m) * 100
        return (m * 60) + s



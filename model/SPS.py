from model.SafeGuard import SafeGuardUtility


class SPS:
    """
    停车点步进机制模拟

    列车状态一旦越过下个辅助停车区的最小速度曲线
    则立即发起步进请求, 并在经过步进冗余时间T_r后
    步进到下个辅助停车区
    """

    def __init__(self, sgu: SafeGuardUtility, numofSPS: int, T_r: float) -> None:
        self.sgu: SafeGuardUtility = sgu  # 进路安全防护实例
        self.IsPrevSPSReqDone: bool = True  # 上次步进请求处理完成标志
        self.SPSReqTimeStamp: float = 0.0  # 步进请求发起时间戳
        self.numofSPS: int = numofSPS  # 进路停车点总数
        self.T_r: float = T_r  # 步进冗余时间

    def StepToNextSP(
        self,
        current_pos: float,
        current_speed: float,
        current_time: float,
        current_sp: int,
    ) -> int:
        """
        根据当前状态发起停车点步进请求

        Args:
            current_pos: 当前位置
            current_speed: 当前速度
            current_time: 当前时间
            current_sp: 当前目标停车点编号

        Returns:
            当前目标停车点
        """
        next_sp = current_sp + 1
        if self.IsPrevSPSReqDone:
            # 上一个步进请求已经执行
            if next_sp <= self.numofSPS - 1:
                # 未步进到最后一个停车点
                # 可以尝试发起步进请求
                (
                    next_min_speed,
                    _,
                ) = self.sgu.GetCurrentMinAndMaxSpeed(
                    current_pos=current_pos,
                    current_sp=next_sp,
                )
                if current_speed > next_min_speed:
                    # 满足步进速度条件，立即发起步进请求
                    self.IsPrevSPSReqDone = False  # 设置标志位
                    self.SPSReqTimeStamp = current_time  # 记录发起步进请求时间戳
            return current_sp
        else:
            if current_time > self.SPSReqTimeStamp + self.T_r:
                # 此时已完成步进
                # 设置执行完成标志
                self.IsPrevSPSReqDone = True
                return next_sp
            else:
                # 未完成步进
                return current_sp

    def Reset(self) -> None:
        """
        重置停车点步进模拟
        """
        self.IsPrevSPSReqDone = True
        self.SPSReqTimeStamp = 0.0

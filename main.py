from configuration.config import parse_arguments_main
from paradigms.centralized.c2d import CentralizedAD2D
# from paradigms.centralized.c3d import CentralizedAD3D  # not implement
# from paradigms.federated.f2d import FederatedAD2D  # not implement


def main(args):
    # centralized learning for 2d anomaly detection
    if args.paradigm == 'c2d':
        work = CentralizedAD2D(args=args)
        work.run_work_flow() # 批量测试

        # todo
        # work.train_small_tool() # 单训练
        # work.init_small_tool(0) # 初始化阶段

    # 以下源代码未实现
    # centralized learning for 3d anomaly detection
    # if args.paradigm == 'c3d':
    #     work = CentralizedAD3D(args=args)
    #     work.run_work_flow()
    # federated learning for 2d anomaly detection
    # if args.paradigm == 'f2d':
    #     work = FederatedAD2D(args=args)
    #     work.run_work_flow()
    

if __name__ == '__main__':
    args = parse_arguments_main()    
    main(args)
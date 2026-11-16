class AnnotationQualityControl:
    """
    标注质量控制
    """
    
    @staticmethod
    def calculate_inter_annotator_agreement(annotations1: List[Dict], 
                                            annotations2: List[Dict]) -> Dict:
        """
        计算两名标注者之间的一致性（Cohen's Kappa）
        """
        from sklearn.metrics import cohen_kappa_score
        
        # 提取转喻类型标签
        y1 = [ann['metonymy_type'] for ann in annotations1]
        y2 = [ann['metonymy_type'] for ann in annotations2]
        
        # 将None转换为字符串
        y1 = [str(y) for y in y1]
        y2 = [str(y) for y in y2]
        
        kappa = cohen_kappa_score(y1, y2)
        
        # 计算各类别的准确率
        types = set(y1 + y2)
        per_type_acc = {}
        for t in types:
            correct = sum(1 for a1, a2 in zip(y1, y2) if a1 == a2 and a1 == t)
            total = sum(1 for a in y1 if a == t)
            per_type_acc[t] = correct / max(total, 1)
        
        return {
            'kappa': kappa,
            'per_type_accuracy': per_type_acc,
            'overall_accuracy': sum(1 for a1, a2 in zip(y1, y2) if a1 == a2) / len(y1)
        }
    
    @staticmethod
    def sample_for_review(df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
        """
        抽取样本进行人工复核
        
        策略：
        1. 优先抽取低置信度的样本
        2. 保证各类转喻都有覆盖
        3. 包含边界案例
        """
        # 低置信度样本
        low_conf = df[df['metonymy_confidence'] < 0.8].sample(
            n=min(n_samples//2, len(df[df['metonymy_confidence'] < 0.8])),
            random_state=42
        )
        
        # 各类转喻样本
        type_samples = []
        for t in ['T1', 'T2', 'T3']:
            type_df = df[df['metonymy_metonymy_type'] == t]
            if len(type_df) > 0:
                type_samples.append(type_df.sample(
                    n=min(n_samples//6, len(type_df)),
                    random_state=42
                ))
        
        # 无转喻样本
        null_df = df[df['metonymy_metonymy_type'].isnull()]
        if len(null_df) > 0:
            type_samples.append(null_df.sample(
                n=min(n_samples//3, len(null_df)),
                random_state=42
            ))
        
        review_df = pd.concat([low_conf] + type_samples, ignore_index=True)
        return review_df.drop_duplicates()
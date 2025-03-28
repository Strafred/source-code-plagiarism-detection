import java.util.Scanner;
public class Main {
    public static double countHeightUser(double feet,double inches){
        return feet*12+inches;
    }
    public static double countBmi (double height,double weight){
        return  weight * 0.45359237 /    (Math.pow((height * 0.0254),2));
    }
    public static void printBmi(double bmi){
        if (bmi < 18.5)
            System.out.println("Underweight");
        else if (bmi > 18.5&& bmi < 25)
            System.out.println("Normal");
        else if (bmi > 25&& bmi < 30)
            System.out.println("Overweight");
        else if(bmi > 30)
            System.out.println("Obese");
    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        double inchesUser,weightUser,feetUser,heightUser,bmi;
        System.out.print("Enter inches: ");
        inchesUser = sc.nextDouble();
        System.out.print("Enter weight in pounds: ");
        weightUser = sc.nextDouble();
        System.out.print("Enter feet: ");
        feetUser = sc.nextDouble();
        heightUser = countHeightUser(feetUser,inchesUser);
        bmi =countBmi(heightUser,weightUser);
        System.out.print("BMI is ");
        System.out.println(bmi);
        printBmi(bmi);
    }
}

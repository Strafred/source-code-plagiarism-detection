import java.util.Scanner;
public class Main {
    public static double countHeightUser(double feet,double inches){
        return feet*12+inches;
    }
    public static double countBmi (double height,double weight){
        return  weight * 0.45359237 /        ((height * 0.0254) * (height * 0.0254));
    }
    public static void printBmi(double bmi){
        if (bmi < 18.5)
            System.out.println("Underweight");
        else if (bmi < 25)
            System.out.println("Normal");
        else if (bmi < 30)
            System.out.println("Overweight");
        else
            System.out.println("Obese");
    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter inches: ");
        double inchesUser = sc.nextDouble();
        System.out.print("Enter weight in pounds: ");
        double weightUser = sc.nextDouble();
        System.out.print("Enter feet: ");
        double feetUser = sc.nextDouble();
        double heightUser = countHeightUser(feetUser,inchesUser);
        double bmi =countBmi(heightUser,weightUser);
        System.out.println("BMI is " + bmi);
        printBmi(bmi);
    }
}

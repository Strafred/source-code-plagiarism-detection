import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter weight in pounds: ");
        double weightUser = sc.nextDouble();
        System.out.print("Enter feet: ");
        double feetUser = sc.nextDouble();
        System.out.print("Enter inches: ");
        double inchesUser = sc.nextDouble();
        double heightUser = feetUser * 12 + inchesUser;
        double bmi = weightUser * 0.45359237 / ((heightUser * 0.0254) * (heightUser * 0.0254));
        System.out.println("BMI is " + bmi);
        if (bmi < 18.5)
            System.out.println("Underweight");
        else if (bmi < 25)
            System.out.println("Normal");
        else if (bmi < 30)
            System.out.println("Overweight");
        else
            System.out.println("Obese");
    }
}
